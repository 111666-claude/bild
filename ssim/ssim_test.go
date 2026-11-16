package ssim

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"
)

// newTestImage returns an opaque texture with both high-frequency detail and
// color variation, so blurring and pixel damage are clearly visible to SSIM.
func newTestImage(rect image.Rectangle) *image.RGBA {
	img := image.NewRGBA(rect)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			ix, iy := x-rect.Min.X, y-rect.Min.Y
			// Checkerboard base plus deterministic color stripes.
			v := 50
			if (ix/2+iy/2)%2 == 0 {
				v = 205
			}
			v = (v + ix*3 + iy*5) % 256
			pos := img.PixOffset(x, y)
			img.Pix[pos+0] = uint8(v)
			img.Pix[pos+1] = uint8((v + ix*7 + 31) % 256)
			img.Pix[pos+2] = uint8((v + iy*11 + 67) % 256)
			img.Pix[pos+3] = 0xFF
		}
	}
	return img
}

func cloneRGBA(src *image.RGBA) *image.RGBA {
	dst := image.NewRGBA(src.Bounds())
	copy(dst.Pix, src.Pix)
	return dst
}

func brightnessShift(src *image.RGBA, delta int) *image.RGBA {
	dst := cloneRGBA(src)
	for i := 0; i < len(dst.Pix); i += 4 {
		for c := 0; c < 3; c++ {
			v := int(dst.Pix[i+c]) + delta
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			dst.Pix[i+c] = uint8(v)
		}
	}
	return dst
}

func damagePatch(src *image.RGBA, rect image.Rectangle) *image.RGBA {
	dst := cloneRGBA(src)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			pos := dst.PixOffset(x, y)
			dst.Pix[pos+0] = 0
			dst.Pix[pos+1] = 0
			dst.Pix[pos+2] = 0
			dst.Pix[pos+3] = 0xFF
		}
	}
	return dst
}

// regionMean returns the average heatmap value inside rect.
func regionMean(heat *image.Gray, rect image.Rectangle) float64 {
	var sum float64
	var n int
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			sum += float64(heat.GrayAt(x, y).Y)
			n++
		}
	}
	return sum / float64(n)
}

func mustCompare(t *testing.T, ref, cand image.Image, opts *Options) *Result {
	t.Helper()
	result, err := Compare(ref, cand, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Heatmap == nil {
		t.Fatal("expected a result with a heatmap")
	}
	return result
}

func TestCompareIdentical(t *testing.T) {
	cases := []struct {
		name string
		img  image.Image
	}{
		{
			name: "color RGBA",
			img:  newTestImage(image.Rect(0, 0, 23, 17)),
		},
		{
			name: "non-zero origin",
			img:  newTestImage(image.Rect(7, 11, 30, 28)),
		},
		{
			name: "gray",
			img: func() image.Image {
				g := image.NewGray(image.Rect(3, 4, 21, 19))
				for y := 4; y < 19; y++ {
					for x := 3; x < 21; x++ {
						g.SetGray(x, y, color.Gray{Y: uint8((x*13 + y*29) % 256)})
					}
				}
				return g
			}(),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var before []uint8
			if src, ok := c.img.(*image.RGBA); ok {
				before = append([]uint8(nil), src.Pix...)
			}
			result := mustCompare(t, c.img, c.img, nil)

			if math.Abs(result.Score-1) > 1e-12 {
				t.Fatalf("expected a score of 1, got %.15f", result.Score)
			}
			if !result.Heatmap.Bounds().Eq(c.img.Bounds()) {
				t.Fatalf("heatmap bounds %v do not match input bounds %v", result.Heatmap.Bounds(), c.img.Bounds())
			}
			for y := result.Heatmap.Bounds().Min.Y; y < result.Heatmap.Bounds().Max.Y; y++ {
				for x := result.Heatmap.Bounds().Min.X; x < result.Heatmap.Bounds().Max.X; x++ {
					if v := result.Heatmap.GrayAt(x, y).Y; v != 0xFF {
						t.Fatalf("expected a perfect heatmap value at (%d,%d), got %d", x, y, v)
					}
				}
			}
			for i := range before {
				if before[i] != c.img.(*image.RGBA).Pix[i] {
					t.Fatal("Compare modified its input image")
				}
			}
		})
	}
}

func TestCompareDifferentOriginsSamePixels(t *testing.T) {
	base := newTestImage(image.Rect(5, 9, 25, 24))
	shifted := image.NewRGBA(image.Rect(0, 0, 20, 15))
	for y := 0; y < 15; y++ {
		for x := 0; x < 20; x++ {
			sp := base.PixOffset(5+x, 9+y)
			dp := shifted.PixOffset(x, y)
			copy(shifted.Pix[dp:dp+4], base.Pix[sp:sp+4])
		}
	}

	result := mustCompare(t, base, shifted, nil)
	if math.Abs(result.Score-1) > 1e-12 {
		t.Fatalf("expected identical pixels with different origins to score 1, got %v", result.Score)
	}
	if !result.Heatmap.Bounds().Eq(base.Bounds()) {
		t.Fatalf("heatmap must follow the reference bounds, got %v", result.Heatmap.Bounds())
	}
}

func TestCompareScoreOrdering(t *testing.T) {
	base := newTestImage(image.Rect(0, 0, 32, 32))

	mildBrightness := brightnessShift(base, 10)
	heavyBrightness := brightnessShift(base, 60)
	mildBlur := blurForTest(base, 1)
	heavyBlur := blurForTest(base, 3)

	identical := mustCompare(t, base, base, nil).Score
	mildBright := mustCompare(t, base, mildBrightness, nil).Score
	heavyBright := mustCompare(t, base, heavyBrightness, nil).Score
	mildBlurScore := mustCompare(t, base, mildBlur, nil).Score
	heavyBlurScore := mustCompare(t, base, heavyBlur, nil).Score

	if !(identical > mildBright) {
		t.Fatalf("identical image (%.6f) should beat a mild brightness shift (%.6f)", identical, mildBright)
	}
	if !(mildBright > heavyBright) {
		t.Fatalf("a mild brightness shift (%.6f) should beat a heavy one (%.6f)", mildBright, heavyBright)
	}
	if !(heavyBright > mildBlurScore) {
		t.Fatalf("a brightness shift (%.6f) should beat mild detail loss (%.6f)", heavyBright, mildBlurScore)
	}
	if !(mildBlurScore > heavyBlurScore) {
		t.Fatalf("mild blur (%.6f) should beat heavy blur (%.6f)", mildBlurScore, heavyBlurScore)
	}
}

// blurForTest is a small separable box blur, kept local to avoid depending on
// the blur package inside its own test.
func blurForTest(src *image.RGBA, radius int) *image.RGBA {
	b := src.Bounds()
	tmp := image.NewRGBA(b)
	dst := image.NewRGBA(b)
	size := 2*radius + 1

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			var r, g, bl int
			for k := -radius; k <= radius; k++ {
				sx := x + k
				if sx < b.Min.X {
					sx = b.Min.X
				} else if sx >= b.Max.X {
					sx = b.Max.X - 1
				}
				p := src.PixOffset(sx, y)
				r += int(src.Pix[p+0])
				g += int(src.Pix[p+1])
				bl += int(src.Pix[p+2])
			}
			p := tmp.PixOffset(x, y)
			tmp.Pix[p+0] = uint8(r / size)
			tmp.Pix[p+1] = uint8(g / size)
			tmp.Pix[p+2] = uint8(bl / size)
			tmp.Pix[p+3] = 0xFF
		}
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			var r, g, bl int
			for k := -radius; k <= radius; k++ {
				sy := y + k
				if sy < b.Min.Y {
					sy = b.Min.Y
				} else if sy >= b.Max.Y {
					sy = b.Max.Y - 1
				}
				p := tmp.PixOffset(x, sy)
				r += int(tmp.Pix[p+0])
				g += int(tmp.Pix[p+1])
				bl += int(tmp.Pix[p+2])
			}
			p := dst.PixOffset(x, y)
			dst.Pix[p+0] = uint8(r / size)
			dst.Pix[p+1] = uint8(g / size)
			dst.Pix[p+2] = uint8(bl / size)
			dst.Pix[p+3] = 0xFF
		}
	}
	return dst
}

func TestCompareLocalDamageHeatmap(t *testing.T) {
	base := newTestImage(image.Rect(0, 0, 40, 40))
	damaged := damagePatch(base, image.Rect(16, 16, 24, 24))

	result := mustCompare(t, base, damaged, nil)
	if result.Score >= 1 {
		t.Fatalf("a damaged patch must lower the score below 1, got %v", result.Score)
	}

	inside := regionMean(result.Heatmap, image.Rect(18, 18, 22, 22))
	outside := regionMean(result.Heatmap, image.Rect(0, 0, 8, 8))
	if inside >= outside {
		t.Fatalf("damaged area (mean %.1f) must have a lower heat value than an intact area (mean %.1f)", inside, outside)
	}
	if inside > 64 {
		t.Fatalf("damaged area should be dark on the heatmap, got mean %.1f", inside)
	}
	if outside < 200 {
		t.Fatalf("intact area should stay bright on the heatmap, got mean %.1f", outside)
	}

	smallDamage := mustCompare(t, base, damagePatch(base, image.Rect(18, 18, 22, 22)), nil).Score
	largeDamage := mustCompare(t, base, damagePatch(base, image.Rect(12, 12, 28, 28)), nil).Score
	if !(smallDamage > largeDamage) {
		t.Fatalf("a small patch (%.6f) should score higher than a large one (%.6f)", smallDamage, largeDamage)
	}
}

func TestCompareWindowOne(t *testing.T) {
	ref := newTestImage(image.Rect(0, 0, 6, 5))
	cand := brightnessShift(ref, 30)
	opts := DefaultOptions()
	opts.WindowSize = 1

	// With a 1x1 window there is no variance, so contrast and structure are 1
	// and every local score is the luminance term.
	c1 := math.Pow(opts.K1*opts.DynamicRange, 2)
	var expected float64
	var count int
	for y := 0; y < 5; y++ {
		for x := 0; x < 6; x++ {
			refPos := ref.PixOffset(x, y)
			candPos := cand.PixOffset(x, y)
			mr := lumaOf(ref.Pix[refPos:])
			mc := lumaOf(cand.Pix[candPos:])
			expected += (2*mr*mc + c1) / (mr*mr + mc*mc + c1)
			count++
		}
	}
	expected /= float64(count)

	result := mustCompare(t, ref, cand, opts)
	if math.Abs(result.Score-expected) > 1e-12 {
		t.Fatalf("expected %.15f for a 1x1 window, got %.15f", expected, result.Score)
	}
	if math.Abs(result.Contrast-1) > 1e-12 || math.Abs(result.Structure-1) > 1e-12 {
		t.Fatalf("constant windows must have contrast and structure 1, got %v / %v", result.Contrast, result.Structure)
	}
}

func lumaOf(pix []uint8) float64 {
	return lumaR*float64(pix[0]) + lumaG*float64(pix[1]) + lumaB*float64(pix[2])
}

func TestCompareConstantImages(t *testing.T) {
	constant := func(v uint8) *image.RGBA {
		img := image.NewRGBA(image.Rect(0, 0, 9, 9))
		for i := 0; i < len(img.Pix); i += 4 {
			img.Pix[i+0] = v
			img.Pix[i+1] = v
			img.Pix[i+2] = v
			img.Pix[i+3] = 0xFF
		}
		return img
	}

	same := mustCompare(t, constant(120), constant(120), nil)
	if math.Abs(same.Score-1) > 1e-12 {
		t.Fatalf("equal constant images must score 1, got %v", same.Score)
	}

	different := mustCompare(t, constant(30), constant(220), nil)
	if math.IsNaN(different.Score) || math.IsInf(different.Score, 0) {
		t.Fatalf("constant images must not produce NaN or Inf, got %v", different.Score)
	}
	if different.Score <= 0 || different.Score >= 1 {
		t.Fatalf("different constant images should score between 0 and 1, got %v", different.Score)
	}
	// Constant images have no contrast/structure difference at all.
	if math.Abs(different.Contrast-1) > 1e-12 || math.Abs(different.Structure-1) > 1e-12 {
		t.Fatalf("expected contrast and structure 1, got %v / %v", different.Contrast, different.Structure)
	}
	if math.Abs(different.Score-different.Luminance) > 1e-12 {
		t.Fatalf("expected score == luminance for constant images, got %v vs %v", different.Score, different.Luminance)
	}

	// Equal black images with a zero luma are the classic division-by-zero
	// case; the stabilizers must keep the result finite.
	black := mustCompare(t, constant(0), constant(0), nil)
	if math.Abs(black.Score-1) > 1e-12 {
		t.Fatalf("equal black images must score 1, got %v", black.Score)
	}
}

func TestCompareTransparentPixelsAreIgnored(t *testing.T) {
	ref := newTestImage(image.Rect(0, 0, 12, 12))
	cand := cloneRGBA(ref)

	// Fully transparent pixels carry no visible color. Give them completely
	// different RGB values in the candidate; they must not affect the score.
	hole := image.Rect(3, 3, 9, 9)
	for y := hole.Min.Y; y < hole.Max.Y; y++ {
		for x := hole.Min.X; x < hole.Max.X; x++ {
			rp := ref.PixOffset(x, y)
			cp := cand.PixOffset(x, y)
			ref.Pix[rp+3] = 0
			cand.Pix[cp+3] = 0
			cand.Pix[cp+0] = 0xFF
			cand.Pix[cp+1] = 0x00
			cand.Pix[cp+2] = 0xFF
		}
	}

	result := mustCompare(t, ref, cand, nil)
	if math.Abs(result.Score-1) > 1e-12 {
		t.Fatalf("differences in fully transparent pixels must be ignored, got %v", result.Score)
	}

	// A fully transparent image has no comparison weight at all; it must not
	// divide by zero and is neutral rather than damaged.
	clearRef := image.NewRGBA(image.Rect(0, 0, 8, 8))
	clearCand := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for i := 3; i < len(clearCand.Pix); i += 4 {
		clearCand.Pix[i-3] = 0xFF
	}
	clearResult := mustCompare(t, clearRef, clearCand, nil)
	if math.Abs(clearResult.Score-1) > 1e-12 {
		t.Fatalf("a fully transparent image must be neutral, got %v", clearResult.Score)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if v := clearResult.Heatmap.GrayAt(x, y).Y; v != 0xFF {
				t.Fatalf("expected neutral heatmap for transparent pixels, got %d", v)
			}
		}
	}
}

func TestComparePartialAlphaWeights(t *testing.T) {
	// NRGBA stores the color independently of alpha, so lowering alpha keeps
	// the visible color and only reduces the sample weight: the score must
	// stay at 1.
	ref := image.NewNRGBA(image.Rect(0, 0, 10, 10))
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			p := ref.PixOffset(x, y)
			ref.Pix[p+0] = uint8((x*17 + y*5) % 256)
			ref.Pix[p+1] = uint8((x*3 + y*23) % 256)
			ref.Pix[p+2] = uint8((x*11 + y*7) % 256)
			ref.Pix[p+3] = 0xFF
		}
	}
	cand := image.NewNRGBA(ref.Bounds())
	copy(cand.Pix, ref.Pix)
	for i := 3; i < len(cand.Pix); i += 4 {
		cand.Pix[i] = 128
	}

	result := mustCompare(t, ref, cand, nil)
	if math.Abs(result.Score-1) > 1e-12 {
		t.Fatalf("changing only alpha must not change the structural score, got %v", result.Score)
	}
}

func TestComparePremultipliedAndNonPremultipliedMatch(t *testing.T) {
	// The same visible NRGBA image expressed as premultiplied RGBA must give
	// the same score. The only fully opaque frame avoids the lossy
	// premultiplication of the conversion itself.
	nrgba := image.NewNRGBA(image.Rect(0, 0, 12, 9))
	rgba := image.NewRGBA(nrgba.Bounds())
	for y := 0; y < 9; y++ {
		for x := 0; x < 12; x++ {
			p := nrgba.PixOffset(x, y)
			v := uint8((x*19 + y*31) % 256)
			nrgba.Pix[p+0] = v
			nrgba.Pix[p+1] = uint8((int(v) + x*7) % 256)
			nrgba.Pix[p+2] = uint8((int(v) + y*13) % 256)
			nrgba.Pix[p+3] = 0xFF

			q := rgba.PixOffset(x, y)
			copy(rgba.Pix[q:q+4], nrgba.Pix[p:p+4])
		}
	}

	result := mustCompare(t, nrgba, rgba, nil)
	if math.Abs(result.Score-1) > 1e-12 {
		t.Fatalf("the same opaque pixels in NRGBA and RGBA must score 1, got %v", result.Score)
	}
}

func TestCompareGrayAndColorInputs(t *testing.T) {
	rgba := newTestImage(image.Rect(0, 0, 14, 10))
	gray := image.NewGray(rgba.Bounds())
	for y := 0; y < 10; y++ {
		for x := 0; x < 14; x++ {
			p := rgba.PixOffset(x, y)
			gray.SetGray(x, y, color.Gray{Y: uint8(lumaOf(rgba.Pix[p:]))})
		}
	}

	result := mustCompare(t, gray, rgba, nil)
	if result.Score <= 0 || result.Score >= 1 {
		t.Fatalf("expected a partial score when comparing luminance against color pixels, got %v", result.Score)
	}
	if !result.Heatmap.Bounds().Eq(gray.Bounds()) {
		t.Fatalf("unexpected heatmap bounds: %v", result.Heatmap.Bounds())
	}
}

func TestCompareWindowLargerThanImage(t *testing.T) {
	ref := newTestImage(image.Rect(0, 0, 5, 4))
	cand := brightnessShift(ref, 20)
	opts := DefaultOptions()
	opts.WindowSize = 11

	// Every neighborhood covers the whole image, so the result must equal the
	// single global score.
	result := mustCompare(t, ref, cand, opts)
	wantScore, wantHeat := naiveCompare(ref, cand, *opts)

	if math.Abs(result.Score-wantScore) > 1e-9 {
		t.Fatalf("got %.12f, want %.12f", result.Score, wantScore)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 5; x++ {
			if got, want := int(result.Heatmap.GrayAt(x, y).Y), int(wantHeat.GrayAt(x, y).Y); got != want {
				t.Fatalf("heatmap differs at (%d,%d): got %d, want %d", x, y, got, want)
			}
		}
	}

	// The same comparison must also work on a 1x1 image.
	tiny := image.NewRGBA(image.Rect(0, 0, 1, 1))
	tiny.Pix[3] = 0xFF
	tinyResult := mustCompare(t, tiny, tiny, nil)
	if math.Abs(tinyResult.Score-1) > 1e-12 {
		t.Fatalf("expected a 1x1 image to compare identically to itself, got %v", tinyResult.Score)
	}
}

func TestCompareSubImageStride(t *testing.T) {
	// A sub-image keeps the parent's stride, so the visible pixels are not
	// contiguous. Compare must use the bounds and stride, not assume a packed
	// layout.
	parent := image.NewNRGBA(image.Rect(0, 0, 30, 24))
	for y := 0; y < 24; y++ {
		for x := 0; x < 30; x++ {
			p := parent.PixOffset(x, y)
			parent.Pix[p+0] = uint8((x*13 + y*7) % 256)
			parent.Pix[p+1] = uint8((x*5 + y*17) % 256)
			parent.Pix[p+2] = uint8((x*11 + y*3) % 256)
			parent.Pix[p+3] = 0xFF
		}
	}

	ref := parent.SubImage(image.Rect(4, 5, 24, 19)).(*image.NRGBA)
	cand := parent.SubImage(image.Rect(4, 5, 24, 19)).(*image.NRGBA)

	result := mustCompare(t, ref, cand, nil)
	if !result.Heatmap.Bounds().Eq(ref.Bounds()) {
		t.Fatalf("heatmap bounds %v do not match sub-image bounds %v", result.Heatmap.Bounds(), ref.Bounds())
	}
	if math.Abs(result.Score-1) > 1e-12 {
		t.Fatalf("a sub-image must match itself, got %v", result.Score)
	}

	// Compare the sub-image against a packed copy: same visible pixels, so the
	// score must still be 1.
	packed := image.NewNRGBA(image.Rect(0, 0, 20, 14))
	for y := 0; y < 14; y++ {
		sp := ref.PixOffset(ref.Bounds().Min.X, ref.Bounds().Min.Y+y)
		copy(packed.Pix[y*packed.Stride:y*packed.Stride+20*4], ref.Pix[sp:sp+20*4])
	}
	packedResult := mustCompare(t, ref, packed, nil)
	if math.Abs(packedResult.Score-1) > 1e-12 {
		t.Fatalf("a sub-image must match a packed copy, got %v", packedResult.Score)
	}
}

func TestCompareErrors(t *testing.T) {
	img := newTestImage(image.Rect(0, 0, 8, 8))
	small := newTestImage(image.Rect(0, 0, 4, 4))
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))

	cases := []struct {
		name string
		ref  image.Image
		cand image.Image
		opts *Options
		want error
	}{
		{name: "nil reference", ref: nil, cand: img, want: ErrNilImage},
		{name: "nil candidate", ref: img, cand: nil, want: ErrNilImage},
		{name: "typed nil reference", ref: (*image.RGBA)(nil), cand: img, want: ErrNilImage},
		{name: "empty reference", ref: empty, cand: img, want: ErrEmptyImage},
		{name: "empty candidate", ref: img, cand: empty, want: ErrEmptyImage},
		{name: "width mismatch", ref: img, cand: small, want: ErrSizeMismatch},
		{
			name: "height mismatch",
			ref:  img,
			cand: newTestImage(image.Rect(0, 0, 8, 5)),
			want: ErrSizeMismatch,
		},
		{
			name: "zero window",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.WindowSize = 0; return o }(),
			want: ErrInvalidWindow,
		},
		{
			name: "negative window",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.WindowSize = -3; return o }(),
			want: ErrInvalidWindow,
		},
		{
			name: "even window",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.WindowSize = 4; return o }(),
			want: ErrInvalidWindow,
		},
		{
			name: "zero-value options",
			ref:  img,
			cand: img,
			opts: &Options{},
			want: ErrInvalidWindow,
		},
		{
			name: "nan k1",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.K1 = math.NaN(); return o }(),
			want: ErrNonFiniteParameter,
		},
		{
			name: "inf k2",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.K2 = math.Inf(1); return o }(),
			want: ErrNonFiniteParameter,
		},
		{
			name: "nan dynamic range",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.DynamicRange = math.NaN(); return o }(),
			want: ErrNonFiniteParameter,
		},
		{
			name: "negative k1",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.K1 = -0.01; return o }(),
			want: ErrInvalidParameter,
		},
		{
			name: "zero k2",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.K2 = 0; return o }(),
			want: ErrInvalidParameter,
		},
		{
			name: "tiny k1 underflows",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.K1 = 1e-200; return o }(),
			want: ErrInvalidParameter,
		},
		{
			name: "huge dynamic range overflows",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.DynamicRange = 1e200; return o }(),
			want: ErrInvalidParameter,
		},
		{
			name: "negative dynamic range",
			ref:  img,
			cand: img,
			opts: func() *Options { o := DefaultOptions(); o.DynamicRange = -255; return o }(),
			want: ErrInvalidParameter,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result, err := Compare(c.ref, c.cand, c.opts)
			if !errors.Is(err, c.want) {
				t.Fatalf("expected error %v, got %v", c.want, err)
			}
			if result != nil {
				t.Fatalf("expected no result on error, got %#v", result)
			}
		})
	}
}

// naiveCompare is a deliberately straightforward implementation of the same
// definition: it rebuilds every clipped window from scratch. It is the oracle
// for the sliding-window implementation.
func naiveCompare(refImg, candImg image.Image, opts Options) (float64, *image.Gray) {
	ref := asShallowNRGBA(refImg)
	cand := asShallowNRGBA(candImg)
	rb := ref.Bounds()
	cb := cand.Bounds()
	w, h := rb.Dx(), rb.Dy()
	radius := opts.WindowSize / 2
	c1 := math.Pow(opts.K1*opts.DynamicRange, 2)
	c2 := math.Pow(opts.K2*opts.DynamicRange, 2)

	heat := image.NewGray(rb)
	var total int
	var score float64

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			x0, x1 := max(0, x-radius), min(w-1, x+radius)
			y0, y1 := max(0, y-radius), min(h-1, y+radius)

			var weight, sumRef, sumCand, sumRefSq, sumCandSq, sumRefCand float64
			for yy := y0; yy <= y1; yy++ {
				for xx := x0; xx <= x1; xx++ {
					rp := ref.PixOffset(rb.Min.X+xx, rb.Min.Y+yy)
					cp := cand.PixOffset(cb.Min.X+xx, cb.Min.Y+yy)

					wt := float64(ref.Pix[rp+3]) / 255 * float64(cand.Pix[cp+3]) / 255
					if wt == 0 {
						continue
					}
					rl := lumaOf(ref.Pix[rp:])
					cl := lumaOf(cand.Pix[cp:])

					weight += wt
					sumRef += wt * rl
					sumCand += wt * cl
					sumRefSq += wt * rl * rl
					sumCandSq += wt * cl * cl
					sumRefCand += wt * rl * cl
				}
			}

			pos := y*heat.Stride + x
			if weight <= 0 {
				heat.Pix[pos] = 0xFF
				continue
			}

			meanRef := sumRef / weight
			meanCand := sumCand / weight
			varRef := math.Max(sumRefSq/weight-meanRef*meanRef, 0)
			varCand := math.Max(sumCandSq/weight-meanCand*meanCand, 0)
			cov := sumRefCand/weight - meanRef*meanCand

			lum := (2*meanRef*meanCand + c1) / (meanRef*meanRef + meanCand*meanCand + c1)
			cs := (2*cov + c2) / (varRef + varCand + c2)
			local := lum * cs

			total++
			score += local
			heat.Pix[pos] = heatValue(local)
		}
	}

	if total == 0 {
		return 1, heat
	}
	return score / float64(total), heat
}

func TestCompareAgainstNaiveImplementation(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	for _, window := range []int{1, 3, 7, 15} {
		t.Run(fmt.Sprintf("window_%d", window), func(t *testing.T) {
			rect := image.Rect(0, 0, 19, 13)
			ref := image.NewRGBA(rect)
			cand := image.NewRGBA(rect)
			for y := 0; y < 13; y++ {
				for x := 0; x < 19; x++ {
					rp := ref.PixOffset(x, y)
					cp := cand.PixOffset(x, y)
					ref.Pix[rp+0] = uint8(rng.Intn(256))
					ref.Pix[rp+1] = uint8(rng.Intn(256))
					ref.Pix[rp+2] = uint8(rng.Intn(256))
					ref.Pix[rp+3] = uint8(rng.Intn(256))

					cand.Pix[cp+0] = uint8(rng.Intn(256))
					cand.Pix[cp+1] = uint8(rng.Intn(256))
					cand.Pix[cp+2] = uint8(rng.Intn(256))
					cand.Pix[cp+3] = uint8(rng.Intn(256))
				}
			}

			opts := DefaultOptions()
			opts.WindowSize = window

			result := mustCompare(t, ref, cand, opts)
			wantScore, wantHeat := naiveCompare(ref, cand, *opts)

			if math.Abs(result.Score-wantScore) > 1e-9 {
				t.Fatalf("score differs from the naive implementation: got %.12f, want %.12f", result.Score, wantScore)
			}
			for y := 0; y < 13; y++ {
				for x := 0; x < 19; x++ {
					got := int(result.Heatmap.GrayAt(x, y).Y)
					want := int(wantHeat.GrayAt(x, y).Y)
					if diff := got - want; diff < -1 || diff > 1 {
						t.Fatalf("heatmap differs at (%d,%d): got %d, want %d", x, y, got, want)
					}
				}
			}
		})
	}
}

func TestCompareNaiveNonZeroOrigin(t *testing.T) {
	ref := newTestImage(image.Rect(4, 7, 23, 20))
	cand := damagePatch(ref, image.Rect(10, 12, 16, 18))
	opts := DefaultOptions()

	result := mustCompare(t, ref, cand, opts)
	wantScore, wantHeat := naiveCompare(ref, cand, *opts)

	if math.Abs(result.Score-wantScore) > 1e-9 {
		t.Fatalf("score differs from the naive implementation: got %.12f, want %.12f", result.Score, wantScore)
	}
	if !result.Heatmap.Bounds().Eq(ref.Bounds()) {
		t.Fatalf("heatmap bounds %v do not match reference bounds %v", result.Heatmap.Bounds(), ref.Bounds())
	}
	for y := ref.Bounds().Min.Y; y < ref.Bounds().Max.Y; y++ {
		for x := ref.Bounds().Min.X; x < ref.Bounds().Max.X; x++ {
			got := int(result.Heatmap.GrayAt(x, y).Y)
			want := int(wantHeat.GrayAt(x, y).Y)
			if diff := got - want; diff < -1 || diff > 1 {
				t.Fatalf("heatmap differs at (%d,%d): got %d, want %d", x, y, got, want)
			}
		}
	}
}

func TestComparePartialAlphaAgainstNaive(t *testing.T) {
	rect := image.Rect(0, 0, 17, 11)
	ref := newTestImage(rect)
	cand := cloneRGBA(ref)

	// Vary the candidate's alpha in a patch so the window weights change
	// without the visible colors changing.
	for y := 3; y < 8; y++ {
		for x := 5; x < 12; x++ {
			pos := cand.PixOffset(x, y)
			cand.Pix[pos+3] = uint8(40 + y*10)
		}
	}

	opts := DefaultOptions()
	result := mustCompare(t, ref, cand, opts)
	wantScore, wantHeat := naiveCompare(ref, cand, *opts)

	if math.Abs(result.Score-wantScore) > 1e-9 {
		t.Fatalf("score differs from the naive implementation: got %.12f, want %.12f", result.Score, wantScore)
	}
	for y := 0; y < 11; y++ {
		for x := 0; x < 17; x++ {
			got := int(result.Heatmap.GrayAt(x, y).Y)
			want := int(wantHeat.GrayAt(x, y).Y)
			if diff := got - want; diff < -1 || diff > 1 {
				t.Fatalf("heatmap differs at (%d,%d): got %d, want %d", x, y, got, want)
			}
		}
	}
}

// TestCompareGoldenValues checks a few cases against values derived by hand
// from the SSIM definition, independent of both the production loop and the
// naive test oracle.
func TestCompareGoldenValues(t *testing.T) {
	const level = 255.0
	c1 := math.Pow(DefaultK1*level, 2)
	c2 := math.Pow(DefaultK2*level, 2)
	_ = c2 // used by the second case

	constant := func(v uint8) *image.Gray {
		img := image.NewGray(image.Rect(0, 0, 5, 4))
		for y := 0; y < 4; y++ {
			for x := 0; x < 5; x++ {
				img.SetGray(x, y, color.Gray{Y: v})
			}
		}
		return img
	}

	// Case 1: two constant images. Every window has zero variance, so the
	// contrast and structure terms are (0 + C2) / (0 + C2) = 1 and the local
	// score is exactly the luminance term. The window shape (clipped or not)
	// cannot change the result.
	refLevel := 100.0
	candLevel := 110.0
	wantLum := (2*refLevel*candLevel + c1) / (refLevel*refLevel + candLevel*candLevel + c1)

	result := mustCompare(t, constant(100), constant(110), nil)
	if math.Abs(result.Score-wantLum) > 1e-12 {
		t.Fatalf("constant images: got score %.15f, want %.15f", result.Score, wantLum)
	}
	if math.Abs(result.Luminance-wantLum) > 1e-12 || math.Abs(result.Contrast-1) > 1e-12 || math.Abs(result.Structure-1) > 1e-12 {
		t.Fatalf("constant images: unexpected terms %+v", *result)
	}
	wantHeat := uint8(math.Round(wantLum * 255))
	for y := 0; y < 4; y++ {
		for x := 0; x < 5; x++ {
			if got := result.Heatmap.GrayAt(x, y).Y; got != wantHeat {
				t.Fatalf("constant heatmap at (%d,%d): got %d, want %d", x, y, got, wantHeat)
			}
		}
	}

	// Case 2: a 2x2 image with a 1x1 window. There is no neighborhood, so
	// every local score is the luminance term of a single pixel, and the mean
	// is their average. Gray pixels have R=G=B, so their luma is the level.
	// With pixels ref = {10, 20, 30, 40} and cand = {40, 30, 20, 10}, laid out
	// in a deliberately asymmetric order.
	refGrid := []uint8{10, 20, 30, 40}
	candGrid := []uint8{40, 30, 20, 10}
	ref := image.NewGray(image.Rect(0, 0, 2, 2))
	cand := image.NewGray(image.Rect(0, 0, 2, 2))
	var want float64
	for i := 0; i < 4; i++ {
		a := float64(refGrid[i])
		b := float64(candGrid[i])
		ref.SetGray(i%2, i/2, color.Gray{Y: refGrid[i]})
		cand.SetGray(i%2, i/2, color.Gray{Y: candGrid[i]})
		want += (2*a*b + c1) / (a*a + b*b + c1)
	}
	want /= 4

	opts := DefaultOptions()
	opts.WindowSize = 1
	single := mustCompare(t, ref, cand, opts)
	if math.Abs(single.Score-want) > 1e-12 {
		t.Fatalf("1x1 windows: got score %.15f, want %.15f", single.Score, want)
	}
	if math.Abs(single.Contrast-1) > 1e-12 || math.Abs(single.Structure-1) > 1e-12 {
		t.Fatalf("1x1 windows must have contrast and structure 1, got %v / %v", single.Contrast, single.Structure)
	}
}

func TestCompareDefaultOptionsIsValid(t *testing.T) {
	opts := DefaultOptions()
	if opts.WindowSize != DefaultWindowSize || opts.K1 != DefaultK1 || opts.K2 != DefaultK2 || opts.DynamicRange != DefaultDynamicRange {
		t.Fatalf("unexpected default options: %+v", opts)
	}
	// Mutating the returned value must not leak into later calls.
	opts.WindowSize = 3
	if DefaultOptions().WindowSize != DefaultWindowSize {
		t.Fatal("DefaultOptions must return a fresh value")
	}
}

func TestScoreShortcut(t *testing.T) {
	img := newTestImage(image.Rect(0, 0, 9, 9))
	score, err := Score(img, img)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.Abs(score-1) > 1e-12 {
		t.Fatalf("expected score 1, got %v", score)
	}

	if _, err := Score(img, newTestImage(image.Rect(0, 0, 4, 4))); !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("expected ErrSizeMismatch, got %v", err)
	}
}

func BenchmarkCompare(b *testing.B) {
	ref := newTestImage(image.Rect(0, 0, 512, 512))
	cand := brightnessShift(ref, 12)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Compare(ref, cand, nil); err != nil {
			b.Fatal(err)
		}
	}
}
