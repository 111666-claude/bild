package blur

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

func TestBilateralInvalidParams(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))

	cases := []struct {
		name                      string
		src                       image.Image
		radius, spatial, colorSig float64
		expected                  error
	}{
		{"nil image", nil, 2, 2, 20, ErrNilImage},
		{"zero radius", img, 0, 2, 20, ErrInvalidRadius},
		{"negative radius", img, -1, 2, 20, ErrInvalidRadius},
		{"NaN radius", img, math.NaN(), 2, 20, ErrInvalidRadius},
		{"infinite radius", img, math.Inf(1), 2, 20, ErrInvalidRadius},
		{"negative infinite radius", img, math.Inf(-1), 2, 20, ErrInvalidRadius},
		{"zero spatial sigma", img, 2, 0, 20, ErrInvalidSpatialSigma},
		{"negative spatial sigma", img, 2, -3, 20, ErrInvalidSpatialSigma},
		{"NaN spatial sigma", img, 2, math.NaN(), 20, ErrInvalidSpatialSigma},
		{"infinite spatial sigma", img, 2, math.Inf(1), 20, ErrInvalidSpatialSigma},
		{"zero color sigma", img, 2, 2, 0, ErrInvalidColorSigma},
		{"negative color sigma", img, 2, 2, -5, ErrInvalidColorSigma},
		{"NaN color sigma", img, 2, 2, math.NaN(), ErrInvalidColorSigma},
		{"infinite color sigma", img, 2, 2, math.Inf(-1), ErrInvalidColorSigma},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, err := Bilateral(c.src, c.radius, c.spatial, c.colorSig)
			if !errors.Is(err, c.expected) {
				t.Fatalf("expected error %v, got %v", c.expected, err)
			}
			if out != nil {
				t.Fatalf("expected nil image on error, got %v", out.Bounds())
			}
		})
	}
}

func TestBilateralBoundsAndInputUnchanged(t *testing.T) {
	bounds := image.Rect(7, 11, 7+9, 11+6)
	src := noisyImage(bounds, 8)
	before := append([]byte(nil), src.Pix...)

	out, err := Bilateral(src, 2, 2, 20)
	if err != nil {
		t.Fatal(err)
	}

	if !out.Bounds().Eq(src.Bounds()) {
		t.Fatalf("expected bounds %v, got %v", src.Bounds(), out.Bounds())
	}
	if !out.Bounds().Eq(bounds) {
		t.Fatalf("expected bounds %v, got %v", bounds, out.Bounds())
	}
	if string(src.Pix) != string(before) {
		t.Fatal("Bilateral modified the source image")
	}
}

func TestBilateralEmptyImage(t *testing.T) {
	bounds := image.Rect(3, 4, 3, 4)
	out, err := Bilateral(image.NewRGBA(bounds), 2, 2, 20)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Bounds().Eq(bounds) {
		t.Fatalf("expected bounds %v, got %v", bounds, out.Bounds())
	}
}

// TestBilateralSmoothsFlatNoise checks that low-amplitude noise on a solid
// color patch is averaged out.
func TestBilateralSmoothsFlatNoise(t *testing.T) {
	src := flatNoisyImage(image.Rect(0, 0, 16, 16), 8)

	out, err := Bilateral(src, 2, 2.0, 25)
	if err != nil {
		t.Fatal(err)
	}

	const margin = 2
	beforeVar := variance(src, margin)
	afterVar := variance(out, margin)

	if afterVar >= beforeVar/4 {
		t.Fatalf("noise was not smoothed: variance before %.3f, after %.3f", beforeVar, afterVar)
	}
	if afterVar > 1.5 {
		t.Fatalf("noise variance after filtering is too high: %.3f", afterVar)
	}
	if mean := channelMean(out, margin); math.Abs(mean-120) > 1.5 {
		t.Fatalf("mean shifted too much: got %.3f, want 120", mean)
	}

	for y := 0; y < out.Bounds().Dy(); y++ {
		for x := 0; x < out.Bounds().Dx(); x++ {
			if out.RGBAAt(x, y).A != 0xFF {
				t.Fatalf("alpha changed for opaque image at (%d,%d)", x, y)
			}
		}
	}
}

// TestBilateralPreservesStepEdge checks that a sharp color step survives the
// filter, both in color and alpha.
func TestBilateralPreservesStepEdge(t *testing.T) {
	const w, h = 9, 3
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x < 4 {
				src.SetRGBA(x, y, color.RGBA{0, 0, 0, 0xFF})
			} else {
				src.SetRGBA(x, y, color.RGBA{0xFF, 0xFF, 0xFF, 0xFF})
			}
		}
	}

	before := append([]byte(nil), src.Pix...)

	out, err := Bilateral(src, 2, 2.0, 10)
	if err != nil {
		t.Fatal(err)
	}

	if !util.RGBAImageEqual(out, src) {
		t.Fatalf("step edge was softened:\ninput:  %v\noutput: %v", before, out.Pix)
	}
}

// TestBilateralAlphaEdgePreserved checks that a large alpha difference is
// down-weighted just like a color difference.
func TestBilateralAlphaEdgePreserved(t *testing.T) {
	const w, h = 9, 3
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x < 4 {
				src.SetRGBA(x, y, color.RGBA{0xFF, 0xFF, 0xFF, 0xFF})
			} else {
				src.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
			}
		}
	}

	out, err := Bilateral(src, 2, 2.0, 10)
	if err != nil {
		t.Fatal(err)
	}

	if !util.RGBAImageEqual(out, src) {
		t.Fatalf("alpha edge was not preserved:\ninput:  %v\noutput: %v", src.Pix, out.Pix)
	}
}

// TestBilateralPremultipliedAlpha checks that transparent pixels do not bleed
// their (meaningless) color into the result and that color and alpha are
// filtered with the same weights, so un-premultiplied hue stays intact.
func TestBilateralPremultipliedAlpha(t *testing.T) {
	const w, h = 12, 4
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x < w/2 {
				src.SetNRGBA(x, y, color.NRGBA{0xFF, 0, 0, 0xFF})
			} else {
				// Fully transparent pixels carrying white RGB data. A naive
				// non-premultiplied implementation would bleed this white.
				src.SetNRGBA(x, y, color.NRGBA{0xFF, 0xFF, 0xFF, 0})
			}
		}
	}

	out, err := Bilateral(src, 2, 3.0, 1000)
	if err != nil {
		t.Fatal(err)
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := out.RGBAAt(x, y)
			if c.R != c.A {
				t.Fatalf("at (%d,%d): red %d and alpha %d should match to keep the hue", x, y, c.R, c.A)
			}
			if c.G != 0 || c.B != 0 {
				t.Fatalf("at (%d,%d): transparent white bled into the result: %v", x, y, c)
			}
			if c.R > c.A || c.G > c.A || c.B > c.A {
				t.Fatalf("at (%d,%d): result is not premultiplied: %v", x, y, c)
			}
		}
	}

	// Fully transparent input pixels must still be fully transparent.
	if c := out.RGBAAt(w-1, 0); c.A != 0 {
		t.Fatalf("expected fully transparent output, got %v", c)
	}
}

// TestBilateralUniformImageUnchanged checks that a constant image (including
// a constant alpha) is a fixed point of the filter.
func TestBilateralUniformImageUnchanged(t *testing.T) {
	const w, h = 7, 5
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetRGBA(x, y, color.RGBA{64, 32, 16, 80})
		}
	}

	out, err := Bilateral(src, 3, 1.5, 10)
	if err != nil {
		t.Fatal(err)
	}

	if !util.RGBAImageEqual(out, src) {
		t.Fatalf("uniform image changed:\ninput:  %v\noutput: %v", src.Pix, out.Pix)
	}
}

// TestBilateralNonZeroBounds checks that non-zero and strided bounds do not
// shift the result compared to the same pixels at the origin.
func TestBilateralNonZeroBounds(t *testing.T) {
	const w, h = 11, 7

	base := noisyImage(image.Rect(0, 0, w, h), 8)
	shifted := &image.RGBA{
		Rect:   image.Rect(10, 20, 10+w, 20+h),
		Stride: w * 4,
		Pix:    append([]byte(nil), base.Pix...),
	}

	baseOut, err := Bilateral(base, 2, 2, 20)
	if err != nil {
		t.Fatal(err)
	}
	shiftedOut, err := Bilateral(shifted, 2, 2, 20)
	if err != nil {
		t.Fatal(err)
	}

	if !shiftedOut.Bounds().Eq(shifted.Bounds()) {
		t.Fatalf("expected bounds %v, got %v", shifted.Bounds(), shiftedOut.Bounds())
	}
	// util.RGBAImageEqual also compares Rect, so compare the pixels relative to
	// each image's origin instead.
	if !pixelsEqualRelative(baseOut, shiftedOut) {
		t.Fatalf("non-zero origin changed the result:\norigin: %v\nshifted: %v", baseOut.Pix, shiftedOut.Pix)
	}

	// A sub-image shares a wider stride and starts at a non-zero origin.
	sub := base.SubImage(image.Rect(2, 1, 2+w-4, 1+h-3)).(*image.RGBA)
	subOut, err := Bilateral(sub, 2, 2, 20)
	if err != nil {
		t.Fatal(err)
	}

	// Compare with the matching region of the full-image result.
	for y := 0; y < sub.Bounds().Dy(); y++ {
		for x := 0; x < sub.Bounds().Dx(); x++ {
			want := baseOut.RGBAAt(sub.Bounds().Min.X+x, sub.Bounds().Min.Y+y)
			got := subOut.RGBAAt(sub.Bounds().Min.X+x, sub.Bounds().Min.Y+y)
			if got != want {
				t.Fatalf("sub-image mismatch at (%d,%d): got %v, want %v", x, y, got, want)
			}
		}
	}
}

// TestBilateralColorModels checks that 8-bit, 16-bit, non-premultiplied and
// grayscale inputs produce the same result as their RGBA equivalent.
func TestBilateralColorModels(t *testing.T) {
	const w, h = 8, 6

	nrgba := image.NewNRGBA(image.Rect(0, 0, w, h))
	nrgba64 := image.NewNRGBA64(image.Rect(0, 0, w, h))
	gray := image.NewGray(image.Rect(0, 0, w, h))
	gray16 := image.NewGray16(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			a := uint8(80 + (x*11+y*7)%176)
			nrgba.SetNRGBA(x, y, color.NRGBA{
				R: uint8((x*29 + y*13) % 256),
				G: uint8((x*53 + y*31) % 256),
				B: uint8((x*17 + y*47) % 256),
				A: a,
			})
			nrgba64.SetNRGBA64(x, y, color.NRGBA64{
				R: uint16((x*29 + y*13) % 256 * 0x101),
				G: uint16((x*53 + y*31) % 256 * 0x101),
				B: uint16((x*17 + y*47) % 256 * 0x101),
				A: uint16(a) * 0x101,
			})
			gray.SetGray(x, y, color.Gray{Y: uint8((x*37 + y*19) % 256)})
			gray16.SetGray16(x, y, color.Gray16{Y: uint16((x*37+y*19)%256) * 0x101})
		}
	}

	cases := []struct {
		name string
		src  image.Image
	}{
		{"NRGBA", nrgba},
		{"NRGBA64", nrgba64},
		{"Gray", gray},
		{"Gray16", gray16},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Build the exact premultiplied RGBA equivalent through image/draw,
			// the same conversion Bilateral has to perform internally.
			equivalent := image.NewRGBA(c.src.Bounds())
			draw.Draw(equivalent, equivalent.Bounds(), c.src, c.src.Bounds().Min, draw.Src)

			got, err := Bilateral(c.src, 2, 2, 20)
			if err != nil {
				t.Fatal(err)
			}
			want, err := Bilateral(equivalent, 2, 2, 20)
			if err != nil {
				t.Fatal(err)
			}
			if !util.RGBAImageEqual(got, want) {
				t.Fatalf("results differ from the RGBA equivalent:\ngot:  %v\nwant: %v", got.Pix, want.Pix)
			}
			if !got.Bounds().Eq(c.src.Bounds()) {
				t.Fatalf("expected bounds %v, got %v", c.src.Bounds(), got.Bounds())
			}
		})
	}
}

// TestBilateralHugeRadius checks that extremely large (but finite) radii are
// clamped instead of overflowing or allocating unbounded memory.
func TestBilateralHugeRadius(t *testing.T) {
	src := noisyImage(image.Rect(0, 0, 6, 4), 8)

	capped, err := Bilateral(src, 6, 2, 20)
	if err != nil {
		t.Fatal(err)
	}
	huge, err := Bilateral(src, 1e300, 2, 20)
	if err != nil {
		t.Fatal(err)
	}
	if !util.RGBAImageEqual(capped, huge) {
		t.Fatalf("huge radius was not clamped to the largest image dimension")
	}
}

func TestBilateralFilterDelegatesToBilateral(t *testing.T) {
	src := noisyImage(image.Rect(5, 6, 5+7, 6+5), 8)

	direct, err := Bilateral(src, 2, 1.5, 15)
	if err != nil {
		t.Fatal(err)
	}
	viaOptions, err := BilateralFilter(src, BilateralOptions{Radius: 2, SpatialSigma: 1.5, ColorSigma: 15})
	if err != nil {
		t.Fatal(err)
	}
	if !util.RGBAImageEqual(direct, viaOptions) {
		t.Fatal("BilateralFilter and Bilateral produced different results")
	}
}

// pixelsEqualRelative compares two images pixel by pixel, allowing different
// origins as long as the sizes and the pixels relative to each origin match.
func pixelsEqualRelative(a, b *image.RGBA) bool {
	if a.Bounds().Dx() != b.Bounds().Dx() || a.Bounds().Dy() != b.Bounds().Dy() {
		return false
	}
	for y := 0; y < a.Bounds().Dy(); y++ {
		for x := 0; x < a.Bounds().Dx(); x++ {
			if a.RGBAAt(a.Bounds().Min.X+x, a.Bounds().Min.Y+y) != b.RGBAAt(b.Bounds().Min.X+x, b.Bounds().Min.Y+y) {
				return false
			}
		}
	}
	return true
}

// noisyImage returns a deterministic pseudo-random RGBA image with opaque alpha.
func noisyImage(bounds image.Rectangle, dim int) *image.RGBA {
	img := image.NewRGBA(bounds)
	w, h := bounds.Dx(), bounds.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(bounds.Min.X+x, bounds.Min.Y+y, color.RGBA{
				R: uint8((x*37 + y*59 + dim*11) % 256),
				G: uint8((x*53 + y*29 + dim*7) % 256),
				B: uint8((x*17 + y*83 + dim*3) % 256),
				A: 0xFF,
			})
		}
	}
	return img
}

// flatNoisyImage returns a gray patch with small deterministic noise around
// the value 120, simulating the speckles on an otherwise flat background.
func flatNoisyImage(bounds image.Rectangle, dim int) *image.RGBA {
	img := image.NewRGBA(bounds)
	w, h := bounds.Dx(), bounds.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(115 + (x*7+y*13+dim*5)%11)
			img.SetRGBA(bounds.Min.X+x, bounds.Min.Y+y, color.RGBA{v, v, v, 0xFF})
		}
	}
	return img
}

// variance returns the luminance variance over the interior of the image,
// skipping a margin of the given size to avoid edge extension effects.
func variance(img *image.RGBA, margin int) float64 {
	b := img.Bounds()
	mean := channelMean(img, margin)
	var sum float64
	var n int
	for y := b.Min.Y + margin; y < b.Max.Y-margin; y++ {
		for x := b.Min.X + margin; x < b.Max.X-margin; x++ {
			c := img.RGBAAt(x, y)
			v := (float64(c.R) + float64(c.G) + float64(c.B)) / 3
			sum += (v - mean) * (v - mean)
			n++
		}
	}
	return sum / float64(n)
}

func channelMean(img *image.RGBA, margin int) float64 {
	b := img.Bounds()
	var sum float64
	var n int
	for y := b.Min.Y + margin; y < b.Max.Y-margin; y++ {
		for x := b.Min.X + margin; x < b.Max.X-margin; x++ {
			c := img.RGBAAt(x, y)
			sum += (float64(c.R) + float64(c.G) + float64(c.B)) / 3
			n++
		}
	}
	return sum / float64(n)
}
