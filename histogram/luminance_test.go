package histogram

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

func TestNewLuminanceHistogram(t *testing.T) {
	// Transparent pixels must not be counted; alpha does not otherwise
	// influence the luminance bin.
	img := &image.RGBA{
		Rect:   image.Rect(0, 0, 2, 2),
		Stride: 2 * 4,
		Pix: []uint8{
			0x0A, 0x14, 0x1E, 0xFF, // lum 180 -> bin 18
			0x00, 0x00, 0x00, 0x00, // transparent, ignored
			0x64, 0x64, 0x64, 0xFF, // lum 1000 -> bin 100
			0xFF, 0xFF, 0xFF, 0x80, // lum 2550 -> bin 255
		},
	}

	hist := NewLuminanceHistogram(img)

	if len(hist.Bins) != 256 {
		t.Fatalf("expected 256 bins, got %d", len(hist.Bins))
	}

	expected := map[int]int{18: 1, 100: 1, 255: 1}
	for bin, count := range expected {
		if hist.Bins[bin] != count {
			t.Errorf("bin %d: expected %d samples, got %d", bin, count, hist.Bins[bin])
		}
	}

	if total := sumBins(hist.Bins); total != 3 {
		t.Errorf("expected 3 counted samples, got %d", total)
	}
}

func TestNewLuminanceHistogramEmpty(t *testing.T) {
	for _, img := range []image.Image{nil, &image.RGBA{}} {
		hist := NewLuminanceHistogram(img)
		if len(hist.Bins) != 256 {
			t.Fatalf("expected 256 bins, got %d", len(hist.Bins))
		}
		if total := sumBins(hist.Bins); total != 0 {
			t.Errorf("expected no samples for empty input, got %d", total)
		}
	}
}

func TestLuminanceLUTStepGray(t *testing.T) {
	// Eight equally populated levels in both images: every source level is
	// matched to the reference level at the same quantile.
	srcLevels := []uint8{0, 32, 64, 96, 128, 160, 192, 224}
	refLevels := []uint8{16, 48, 80, 112, 144, 176, 208, 240}
	src := stepGrayImage(image.Rect(0, 0, 32, 1), srcLevels, 4)
	ref := stepGrayImage(image.Rect(0, 0, 32, 1), refLevels, 4)

	lut := LuminanceLUT(src, ref)

	if len(lut) != 256 {
		t.Fatalf("expected a 256 entry LUT, got %d", len(lut))
	}

	for i := 1; i < len(lut); i++ {
		if lut[i] < lut[i-1] {
			t.Fatalf("LUT is not monotonic at %d: %d -> %d", i, lut[i-1], lut[i])
		}
	}

	for i, level := range srcLevels {
		if lut[level] != refLevels[i] {
			t.Errorf("LUT[%d]: expected %d, got %d", level, refLevels[i], lut[level])
		}
	}

	// Levels between populated bins resolve to the previous reference level.
	if lut[0] != 16 || lut[31] != 16 {
		t.Errorf("expected levels 0..31 to map to 16, got %d and %d", lut[0], lut[31])
	}
	if lut[255] != 240 {
		t.Errorf("expected level 255 to map to 240, got %d", lut[255])
	}
}

func TestLuminanceLUTIgnoresTransparentPixels(t *testing.T) {
	srcLevels := []uint8{0, 85, 170, 255}
	withTransparent := stepGrayImage(image.Rect(0, 0, 4, 2), srcLevels, 1)
	// The second row is fully transparent but colored, so it must not change
	// the luminance distribution.
	for pos := withTransparent.Stride; pos < len(withTransparent.Pix); pos += 4 {
		withTransparent.Pix[pos+0] = 0xFF
		withTransparent.Pix[pos+1] = 0x00
		withTransparent.Pix[pos+2] = 0xFF
		withTransparent.Pix[pos+3] = 0x00
	}

	opaque := stepGrayImage(image.Rect(0, 0, 4, 1), srcLevels, 1)
	ref := stepGrayImage(image.Rect(0, 0, 4, 1), []uint8{16, 48, 240, 240}, 1)

	got := LuminanceLUT(withTransparent, ref)
	want := LuminanceLUT(opaque, ref)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LUT[%d]: transparent pixels changed the mapping: expected %d, got %d", i, want[i], got[i])
		}
	}
}

func TestLuminanceLUTDegenerate(t *testing.T) {
	ref := stepGrayImage(image.Rect(0, 0, 4, 1), []uint8{0, 85, 170, 255}, 1)
	allTransparent := image.NewRGBA(image.Rect(0, 0, 3, 3))

	cases := []struct {
		name string
		src  image.Image
		ref  image.Image
	}{
		{name: "nil source", src: nil, ref: ref},
		{name: "nil reference", src: ref, ref: nil},
		{name: "empty source", src: &image.RGBA{}, ref: ref},
		{name: "empty reference", src: ref, ref: &image.RGBA{}},
		{name: "transparent source", src: allTransparent, ref: ref},
		{name: "transparent reference", src: ref, ref: allTransparent},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			lut := LuminanceLUT(c.src, c.ref)
			if len(lut) != 256 {
				t.Fatalf("expected a 256 entry LUT, got %d", len(lut))
			}
			for i, v := range lut {
				if v != uint8(i) {
					t.Fatalf("expected identity mapping, LUT[%d] = %d", i, v)
				}
			}
		})
	}
}

func TestMatchLuminanceStepGray(t *testing.T) {
	srcLevels := []uint8{0, 32, 64, 96, 128, 160, 192, 224}
	refLevels := []uint8{16, 48, 80, 112, 144, 176, 208, 240}
	src := stepGrayImage(image.Rect(0, 0, 32, 1), srcLevels, 4)
	ref := stepGrayImage(image.Rect(0, 0, 32, 1), refLevels, 4)

	out := MatchLuminance(src, ref)

	if !out.Bounds().Eq(src.Bounds()) {
		t.Fatalf("expected bounds %v, got %v", src.Bounds(), out.Bounds())
	}

	// The mapped levels are non-decreasing and stay gray.
	previous := -1
	for x := 0; x < out.Bounds().Dx(); x++ {
		r, g, b, a := out.At(x, 0).RGBA()
		if r != g || g != b {
			t.Fatalf("pixel %d is no longer gray: %d %d %d", x, r>>8, g>>8, b>>8)
		}
		if a>>8 != 0xFF {
			t.Fatalf("pixel %d alpha changed: %d", x, a>>8)
		}
		level := int(r >> 8)
		if level < previous {
			t.Fatalf("mapped levels are not monotonic at pixel %d: %d after %d", x, level, previous)
		}
		previous = level
	}

	// Boundary pixels.
	if got := out.Pix[0]; got != 16 {
		t.Errorf("first pixel: expected 16, got %d", got)
	}
	if got := out.Pix[len(out.Pix)-4]; got != 240 {
		t.Errorf("last pixel: expected 240, got %d", got)
	}

	// The output distribution equals the reference distribution.
	gotHist := NewLuminanceHistogram(out)
	wantHist := NewLuminanceHistogram(ref)
	for i := range wantHist.Bins {
		if gotHist.Bins[i] != wantHist.Bins[i] {
			t.Fatalf("bin %d: expected %d samples, got %d", i, wantHist.Bins[i], gotHist.Bins[i])
		}
	}
}

func TestMatchLuminanceColorAndAlpha(t *testing.T) {
	// Three visible source pixels with luminances in bins 10, 96 and 125.
	// Matching to a reference with one pixel in each of bins 50, 150 and 250
	// maps those bins to 50, 150 and 250 respectively.
	src := &image.RGBA{
		Rect:   image.Rect(0, 0, 4, 1),
		Stride: 4 * 4,
		Pix: []uint8{
			0x3C, 0x78, 0x3C, 0xFF, // (60, 120, 60), lum 960 -> bin 96
			0xC8, 0x64, 0x32, 0x80, // (200, 100, 50), lum 1250 -> bin 125
			0x1E, 0x1E, 0x1E, 0x00, // transparent, untouched
			0x0A, 0x0A, 0x0A, 0xFF, // (10, 10, 10), lum 100 -> bin 10
		},
	}
	ref := stepGrayImage(image.Rect(0, 0, 3, 1), []uint8{50, 150, 250}, 1)

	out := MatchLuminance(src, ref)

	expected := &image.RGBA{
		Rect:   image.Rect(0, 0, 4, 1),
		Stride: 4 * 4,
		Pix: []uint8{
			0x7E, 0xA6, 0x7E, 0xFF, // lightened from lum 960 towards white
			0xFD, 0xF9, 0xF7, 0x80, // lightened from lum 1250 towards white
			0x1E, 0x1E, 0x1E, 0x00, // transparent pixel unchanged
			0x32, 0x32, 0x32, 0xFF, // lightened from lum 100 to 50
		},
	}

	if !util.RGBAImageEqual(out, expected) {
		t.Fatalf("unexpected result:\nexpected:\n%v\nactual:\n%v",
			util.RGBAToString(expected), util.RGBAToString(out))
	}

	// Hue is preserved for the partially transparent color pixel. Near-white
	// quantization allows a small tolerance.
	hueBefore, _, _ := util.RGBToHSL(colorAt(src, 4))
	hueAfter, _, _ := util.RGBToHSL(colorAt(out, 4))
	if d := hueDistance(hueBefore, hueAfter); d > 5 {
		t.Errorf("pixel at offset 4: hue changed by %.2f degrees", d)
	}

	// Transparency is always preserved.
	for pos := 0; pos < len(src.Pix); pos += 4 {
		if out.Pix[pos+3] != src.Pix[pos+3] {
			t.Errorf("pixel at offset %d: alpha changed from %d to %d",
				pos, src.Pix[pos+3], out.Pix[pos+3])
		}
	}
}

func TestMatchLuminanceInputUnchanged(t *testing.T) {
	src := stepGrayImage(image.Rect(0, 0, 8, 1), []uint8{0, 32, 64, 96, 128, 160, 192, 224}, 1)
	ref := stepGrayImage(image.Rect(0, 0, 8, 1), []uint8{16, 48, 80, 112, 144, 176, 208, 240}, 1)

	srcPix := append([]uint8(nil), src.Pix...)
	refPix := append([]uint8(nil), ref.Pix...)

	MatchLuminance(src, ref)

	if !bytesEqual(src.Pix, srcPix) {
		t.Error("source image was modified")
	}
	if !bytesEqual(ref.Pix, refPix) {
		t.Error("reference image was modified")
	}
}

func TestMatchLuminanceNonZeroBounds(t *testing.T) {
	srcLevels := []uint8{0, 32, 64, 96, 128, 160, 192, 224}
	rect := image.Rect(5, 7, 13, 8)
	src := stepGrayImage(rect, srcLevels, 1)
	ref := stepGrayImage(image.Rect(0, 0, 8, 1), srcLevels, 1)

	out := MatchLuminance(src, ref)

	if !out.Bounds().Eq(rect) {
		t.Fatalf("expected bounds %v, got %v", rect, out.Bounds())
	}

	// The same pixels with a zero origin must produce the same data.
	flatSrc := stepGrayImage(image.Rect(0, 0, 8, 1), srcLevels, 1)
	flatOut := MatchLuminance(flatSrc, ref)

	// The bounds differ by design, so compare the pixel data itself.
	if !bytesEqual(out.Pix, flatOut.Pix) {
		t.Error("non-zero origin changed the processed pixel data")
	}
}

func TestMatchLuminanceColorModelsAgree(t *testing.T) {
	levels := []uint8{0, 32, 64, 96, 128, 160, 192, 224}
	gray := stepGrayImage(image.Rect(0, 0, 8, 1), levels, 1)

	// The same grays expressed with other color models and an opaque alpha.
	rgba := image.NewRGBA(image.Rect(0, 0, 8, 1))
	nrgba := image.NewNRGBA(image.Rect(0, 0, 8, 1))
	grayImg := image.NewGray(image.Rect(0, 0, 8, 1))
	gray16 := image.NewGray16(image.Rect(0, 0, 8, 1))
	for x, level := range levels {
		rgba.Set(x, 0, color.RGBA{level, level, level, 0xFF})
		nrgba.Set(x, 0, color.NRGBA{level, level, level, 0xFF})
		grayImg.Set(x, 0, color.Gray{Y: level})
		gray16.Set(x, 0, color.Gray16{Y: uint16(level) * 0x101})
	}

	refLevels := []uint8{16, 48, 80, 112, 144, 176, 208, 240}
	ref := image.NewRGBA(image.Rect(0, 0, 8, 1))
	for x, level := range refLevels {
		ref.Set(x, 0, color.RGBA{level, level, level, 0xFF})
	}

	grayOut := MatchLuminance(gray, ref)
	rgbaOut := MatchLuminance(rgba, ref)
	nrgbaOut := MatchLuminance(nrgba, ref)

	if !util.RGBAImageEqual(grayOut, rgbaOut) {
		t.Error("grayscale and RGBA inputs produced different results")
	}
	if !util.RGBAImageEqual(rgbaOut, nrgbaOut) {
		t.Error("RGBA and NRGBA inputs produced different results")
	}
	if !util.RGBAImageEqual(grayOut, MatchLuminance(grayImg, ref)) {
		t.Error("RGBA and image.Gray inputs produced different results")
	}
	if !util.RGBAImageEqual(grayOut, MatchLuminance(gray16, ref)) {
		t.Error("RGBA and image.Gray16 inputs produced different results")
	}

	// A paletted image goes through the same conversion path.
	paletted := image.NewPaletted(image.Rect(0, 0, 8, 1), color.Palette{
		color.RGBA{0, 0, 0, 0xFF},
		color.RGBA{32, 32, 32, 0xFF},
		color.RGBA{64, 64, 64, 0xFF},
		color.RGBA{96, 96, 96, 0xFF},
		color.RGBA{128, 128, 128, 0xFF},
		color.RGBA{160, 160, 160, 0xFF},
		color.RGBA{192, 192, 192, 0xFF},
		color.RGBA{224, 224, 224, 0xFF},
	})
	for x := range levels {
		paletted.SetColorIndex(x, 0, uint8(x))
	}
	if !util.RGBAImageEqual(grayOut, MatchLuminance(paletted, ref)) {
		t.Error("RGBA and paletted inputs produced different results")
	}
}

func TestMatchLuminanceApproachesReferenceDistribution(t *testing.T) {
	// A full 256-level source matched to a bell-shaped reference. The matched
	// cumulative distribution must track the reference closely, while the
	// source is far from it.
	src := image.NewRGBA(image.Rect(0, 0, 1024, 1))
	for x := 0; x < 1024; x++ {
		level := uint8(x / 4)
		src.Set(x, 0, color.RGBA{level, level, level, 0xFF})
	}
	ref := image.NewRGBA(image.Rect(0, 0, 1024, 1))
	for x := 0; x < 1024; x++ {
		level := uint8(128 + 60*math.Sin(float64(x)/23.0))
		ref.Set(x, 0, color.RGBA{level, level, level, 0xFF})
	}

	out := MatchLuminance(src, ref)

	matchedDev := maxCDFDeviation(NewLuminanceHistogram(out), NewLuminanceHistogram(ref))
	sourceDev := maxCDFDeviation(NewLuminanceHistogram(src), NewLuminanceHistogram(ref))

	if matchedDev > 0.01 {
		t.Errorf("matched CDF deviates from the reference by %.4f, expected <= 0.01", matchedDev)
	}
	if sourceDev <= matchedDev {
		t.Errorf("source CDF deviation %.4f is not improved on by the match (%.4f)", sourceDev, matchedDev)
	}
}

func TestMatchLuminanceDegenerate(t *testing.T) {
	ref := stepGrayImage(image.Rect(0, 0, 4, 1), []uint8{0, 85, 170, 255}, 1)
	transparentRef := image.NewRGBA(image.Rect(0, 0, 4, 1))

	t.Run("empty source", func(t *testing.T) {
		out := MatchLuminance(&image.RGBA{}, ref)
		if !out.Bounds().Empty() {
			t.Errorf("expected empty bounds, got %v", out.Bounds())
		}
	})

	t.Run("nil source", func(t *testing.T) {
		out := MatchLuminance(nil, ref)
		if !out.Bounds().Empty() {
			t.Errorf("expected empty bounds, got %v", out.Bounds())
		}
	})

	t.Run("fully transparent source", func(t *testing.T) {
		src := &image.RGBA{
			Rect:   image.Rect(0, 0, 2, 1),
			Stride: 8,
			Pix:    []uint8{0xFF, 0x00, 0xFF, 0x00, 0x10, 0x20, 0x30, 0x00},
		}
		out := MatchLuminance(src, ref)
		if !util.RGBAImageEqual(out, src) {
			t.Errorf("expected transparent source to be copied unchanged, got:\n%v", util.RGBAToString(out))
		}
	})

	t.Run("fully transparent reference", func(t *testing.T) {
		src := stepGrayImage(image.Rect(0, 0, 4, 1), []uint8{0, 85, 170, 255}, 1)
		out := MatchLuminance(src, transparentRef)
		if !util.RGBAImageEqual(out, src) {
			t.Errorf("expected source to be copied unchanged, got:\n%v", util.RGBAToString(out))
		}
	})

	t.Run("nil reference", func(t *testing.T) {
		src := stepGrayImage(image.Rect(0, 0, 4, 1), []uint8{0, 85, 170, 255}, 1)
		out := MatchLuminance(src, nil)
		if !util.RGBAImageEqual(out, src) {
			t.Errorf("expected source to be copied unchanged, got:\n%v", util.RGBAToString(out))
		}
	})

	t.Run("single color source", func(t *testing.T) {
		src := image.NewRGBA(image.Rect(0, 0, 4, 4))
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				src.Set(x, y, color.RGBA{77, 77, 77, 0xFF})
			}
		}
		out := MatchLuminance(src, ref)
		first := out.Pix[0]
		for pos := 0; pos < len(out.Pix); pos += 4 {
			if out.Pix[pos] != first || out.Pix[pos+1] != first || out.Pix[pos+2] != first {
				t.Fatalf("expected a uniform result, got pixel offset %d", pos)
			}
			if out.Pix[pos+3] != 0xFF {
				t.Fatalf("alpha changed at offset %d", pos)
			}
		}
		// Gray 77 is the sole source level, so the whole image moves to the
		// median level of the reference (85).
		if first != 85 {
			t.Errorf("expected the single level to map to 85, got %d", first)
		}
	})

	t.Run("black source to white reference", func(t *testing.T) {
		src := image.NewRGBA(image.Rect(0, 0, 2, 2))
		for i := 3; i < len(src.Pix); i += 4 {
			src.Pix[i] = 0xFF
		}
		white := image.NewRGBA(image.Rect(0, 0, 2, 2))
		for i := 0; i < len(white.Pix); i++ {
			white.Pix[i] = 0xFF
		}

		out := MatchLuminance(src, white)
		for pos := 0; pos < len(out.Pix); pos += 4 {
			if out.Pix[pos] != 0xFF || out.Pix[pos+1] != 0xFF || out.Pix[pos+2] != 0xFF {
				t.Fatalf("expected white result at offset %d, got %v", pos, out.Pix[pos:pos+4])
			}
		}
	})

	t.Run("white source to black reference", func(t *testing.T) {
		src := image.NewRGBA(image.Rect(0, 0, 2, 2))
		for i := 0; i < len(src.Pix); i++ {
			src.Pix[i] = 0xFF
		}
		black := image.NewRGBA(image.Rect(0, 0, 2, 2))
		for i := 3; i < len(black.Pix); i += 4 {
			black.Pix[i] = 0xFF
		}

		out := MatchLuminance(src, black)
		for pos := 0; pos < len(out.Pix); pos += 4 {
			if out.Pix[pos] != 0x00 || out.Pix[pos+1] != 0x00 || out.Pix[pos+2] != 0x00 {
				t.Fatalf("expected black result at offset %d, got %v", pos, out.Pix[pos:pos+4])
			}
			if out.Pix[pos+3] != 0xFF {
				t.Fatalf("alpha changed at offset %d", pos)
			}
		}
	})
}

func TestMatchLuminanceFollowsLUT(t *testing.T) {
	// Saturated colors exercise the hue-preserving scaling in both
	// directions. Every output luminance must land on the LUT target.
	src := &image.RGBA{
		Rect:   image.Rect(0, 0, 6, 1),
		Stride: 6 * 4,
		Pix: []uint8{
			0xFF, 0x00, 0x00, 0xFF, // red, lum 765
			0x00, 0xFF, 0x00, 0xFF, // green, lum 1530
			0x00, 0x00, 0xFF, 0xFF, // blue, lum 255
			0x80, 0x40, 0x20, 0xFF, // dark warm, lum 500
			0x20, 0x40, 0x80, 0xFF, // dark cool, lum 380
			0xF0, 0xE0, 0x10, 0xFF, // yellow, lum 1210
		},
	}
	ref := stepGrayImage(image.Rect(0, 0, 16, 1),
		[]uint8{0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255}, 1)

	out := MatchLuminance(src, ref)
	lut := LuminanceLUT(src, ref)

	for pos := 0; pos < len(src.Pix); pos += 4 {
		c := colorAt(src, pos)
		lum := luminanceR*int(c.R) + luminanceG*int(c.G) + luminanceB*int(c.B)
		want := lookupLuminance(lut, lum)

		o := colorAt(out, pos)
		got := luminanceR*int(o.R) + luminanceG*int(o.G) + luminanceB*int(o.B)
		if d := got - want; d < -5 || d > 5 {
			t.Errorf("pixel at offset %d: luminance %d, expected %d (tolerance 5)", pos, got, want)
		}
	}
}

func TestMatchLuminanceMonotonicEndToEnd(t *testing.T) {
	// A step-gray source matched to a dark reference stays sorted: brighter
	// source pixels never become darker than dimmer ones.
	srcLevels := []uint8{0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240}
	src := stepGrayImage(image.Rect(0, 0, 32, 1), srcLevels, 2)
	ref := stepGrayImage(image.Rect(0, 0, 6, 1), []uint8{0, 20, 40, 45, 50, 60}, 1)

	out := MatchLuminance(src, ref)

	previous := -1
	for x := 0; x < out.Bounds().Dx(); x++ {
		level := int(out.Pix[x*4])
		if level < previous {
			t.Fatalf("output is not monotonic at pixel %d: %d after %d", x, level, previous)
		}
		previous = level
	}
}

func TestMatchLuminanceIdentity(t *testing.T) {
	src := stepGrayImage(image.Rect(0, 0, 8, 1), []uint8{0, 32, 64, 96, 128, 160, 192, 224}, 1)

	out := MatchLuminance(src, src)

	if !util.RGBAImageEqual(out, src) {
		t.Errorf("expected matching an image to itself to be a no-op, got:\n%v", util.RGBAToString(out))
	}
}

func BenchmarkMatchLuminance(b *testing.B) {
	const size = 256
	src := image.NewRGBA(image.Rect(0, 0, size, size))
	ref := image.NewRGBA(image.Rect(0, 0, size, size))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i+0] = uint8(i % 256)
		src.Pix[i+1] = uint8((i * 3) % 256)
		src.Pix[i+2] = uint8((i * 7) % 256)
		src.Pix[i+3] = 0xFF
		ref.Pix[i+0] = 0x40
		ref.Pix[i+1] = 0x80
		ref.Pix[i+2] = 0xC0
		ref.Pix[i+3] = 0xFF
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatchLuminance(src, ref)
	}
}

// stepGrayImage returns a row-major image in which every level of levels is
// repeated repeat times before moving on to the next level.
func stepGrayImage(rect image.Rectangle, levels []uint8, repeat int) *image.RGBA {
	img := image.NewRGBA(rect)
	pos := 0
	for y := 0; y < rect.Dy(); y++ {
		for _, level := range levels {
			for k := 0; k < repeat; k++ {
				img.Pix[pos+0] = level
				img.Pix[pos+1] = level
				img.Pix[pos+2] = level
				img.Pix[pos+3] = 0xFF
				pos += 4
			}
		}
	}
	return img
}

func colorAt(img *image.RGBA, pos int) color.RGBA {
	return color.RGBA{img.Pix[pos], img.Pix[pos+1], img.Pix[pos+2], img.Pix[pos+3]}
}

func hueDistance(a, b float64) float64 {
	d := a - b
	if d < 0 {
		d = -d
	}
	if d > 180 {
		return 360 - d
	}
	return d
}

// maxCDFDeviation returns the largest absolute difference between the
// normalized cumulative histograms of a and b.
func maxCDFDeviation(a, b *Histogram) float64 {
	ca, cb := normalizedCDF(a), normalizedCDF(b)
	max := 0.0
	for i := range ca {
		d := math.Abs(ca[i] - cb[i])
		if d > max {
			max = d
		}
	}
	return max
}

func normalizedCDF(h *Histogram) []float64 {
	total := sumBins(h.Bins)
	out := make([]float64, len(h.Bins))
	if total == 0 {
		return out
	}
	cumulative := 0
	for i, count := range h.Bins {
		cumulative += count
		out[i] = float64(cumulative) / float64(total)
	}
	return out
}

func sumBins(bins []int) int {
	total := 0
	for _, count := range bins {
		total += count
	}
	return total
}

func bytesEqual(a, b []uint8) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
