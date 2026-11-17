package blur

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

func newRGBA(w, h int, pix []uint8) *image.RGBA {
	return &image.RGBA{
		Rect:   image.Rect(0, 0, w, h),
		Stride: w * 4,
		Pix:    pix,
	}
}

// Invalid parameters must return an unchanged RGBA copy.
func TestBilateralInvalidParameters(t *testing.T) {
	src := newRGBA(2, 2, []uint8{
		0x10, 0x20, 0x30, 0xFF, 0x40, 0x50, 0x60, 0xFF,
		0x70, 0x80, 0x90, 0xFF, 0xA0, 0xB0, 0xC0, 0xFF,
	})

	cases := []struct {
		radius, spatialSigma, colorSigma float64
	}{
		{0, 10, 10},
		{-1, 10, 10},
		{1, 0, 10},
		{1, -5, 10},
		{1, 10, 0},
		{1, 10, -5},
		{math.NaN(), 10, 10},
		{1, math.NaN(), 10},
		{1, 10, math.NaN()},
		{math.Inf(1), 10, 10},
		{1, math.Inf(1), 10},
		{1, 10, math.Inf(1)},
		{1, math.Inf(-1), 10},
	}

	for i, c := range cases {
		actual := Bilateral(src, c.radius, c.spatialSigma, c.colorSigma)
		if !util.RGBAImageEqual(actual, src) {
			t.Errorf("case %d: invalid parameters should return the image unchanged, got %#v", i, actual)
		}
	}
}

// Empty images must return an empty RGBA image without panicking.
func TestBilateralEmptyImage(t *testing.T) {
	actual := Bilateral(image.NewRGBA(image.Rect(0, 0, 0, 0)), 2, 5, 50)
	if !actual.Bounds().Empty() {
		t.Errorf("expected empty bounds, got %v", actual.Bounds())
	}
}

// A constant image must be invariant under any valid parameterization.
func TestBilateralConstantImageInvariant(t *testing.T) {
	cases := []struct {
		radius, spatialSigma, colorSigma float64
	}{
		{1, 1, 1},
		{1, 10, 100},
		{2, 0.5, 500},
	}

	colors := [][4]uint8{
		{0x00, 0x00, 0x00, 0xFF},
		{0xFF, 0xFF, 0xFF, 0xFF},
		{0x80, 0x40, 0x20, 0xFF},
	}

	for _, col := range colors {
		pix := make([]uint8, 4*4*4)
		for i := 0; i < len(pix); i += 4 {
			pix[i+0] = col[0]
			pix[i+1] = col[1]
			pix[i+2] = col[2]
			pix[i+3] = col[3]
		}
		src := newRGBA(4, 4, pix)

		for _, c := range cases {
			actual := Bilateral(src, c.radius, c.spatialSigma, c.colorSigma)
			if !util.RGBAImageEqual(actual, src) {
				t.Errorf("constant image %v with params %v must remain unchanged, got %#v", col, c, actual)
			}
		}
	}
}

// A single pixel image is trivially invariant.
func TestBilateralSinglePixel(t *testing.T) {
	src := newRGBA(1, 1, []uint8{0x12, 0x34, 0x56, 0xFF})
	actual := Bilateral(src, 3, 2, 30)
	if !util.RGBAImageEqual(actual, src) {
		t.Errorf("single pixel image should be unchanged, got %#v", actual)
	}
}

// Hard edges must be preserved: with a tight color sigma, pixels on either
// side of a black/white boundary keep their original values.
func TestBilateralPreservesHardEdges(t *testing.T) {
	// 5x5 image: left half black, right half white, fully opaque.
	pix := make([]uint8, 5*5*4)
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			pos := (y*5 + x) * 4
			if x >= 3 {
				pix[pos+0] = 0xFF
				pix[pos+1] = 0xFF
				pix[pos+2] = 0xFF
			}
			pix[pos+3] = 0xFF
		}
	}
	src := newRGBA(5, 5, pix)

	actual := Bilateral(src, 2, 2.0, 5.0)

	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			pos := (y*5 + x) * 4
			got := actual.Pix[pos]
			want := src.Pix[pos]
			// Edges may only shift by at most one quantization level because
			// cross-edge color weights are effectively zero.
			if d := int(got) - int(want); d < -1 || d > 1 {
				t.Errorf("edge pixel (%d,%d) changed from %d to %d: hard edge was not preserved", x, y, want, got)
			}
			if actual.Pix[pos+3] != 0xFF {
				t.Errorf("alpha changed at (%d,%d): %d", x, y, actual.Pix[pos+3])
			}
		}
	}
}

// Fixed bimodal noise on a flat region must be smoothed toward the local mean,
// while the image content actually changes (proving the filter runs).
func TestBilateralSmoothsFixedNoise(t *testing.T) {
	// Deterministic 5x5 checkerboard of 0 and 255 (mean 132.6 over 25 px).
	pix := make([]uint8, 5*5*4)
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			pos := (y*5 + x) * 4
			if (x+y)%2 == 0 {
				pix[pos+0] = 0xFF
				pix[pos+1] = 0xFF
				pix[pos+2] = 0xFF
			}
			pix[pos+3] = 0xFF
		}
	}
	src := newRGBA(5, 5, pix)

	// Large color sigma turns the color term nearly uniform, so interior
	// pixels converge toward the window mean instead of staying bimodal.
	actual := Bilateral(src, 2, 5.0, 1000.0)

	if util.RGBAImageEqual(actual, src) {
		t.Errorf("noise pattern was not smoothed at all")
	}

	for y := 1; y < 4; y++ {
		for x := 1; x < 4; x++ {
			pos := (y*5 + x) * 4
			got := int(actual.Pix[pos])
			if got < 90 || got > 200 {
				t.Errorf("interior pixel (%d,%d) = %d, expected a mid-tone near the local mean", x, y, got)
			}
		}
	}
}

// Fully transparent neighbors must not pollute the color of visible pixels,
// and transparent pixels must remain transparent.
func TestBilateralAlphaIsolation(t *testing.T) {
	// 3x1: transparent black, opaque red, transparent black.
	src := newRGBA(3, 1, []uint8{
		0x00, 0x00, 0x00, 0x00,
		0xFF, 0x00, 0x00, 0xFF,
		0x00, 0x00, 0x00, 0x00,
	})

	actual := Bilateral(src, 2, 2.0, 30.0)

	center := 4
	if actual.Pix[center+0] != 0xFF || actual.Pix[center+1] != 0 || actual.Pix[center+2] != 0 || actual.Pix[center+3] != 0xFF {
		t.Errorf("opaque pixel polluted by transparent neighbors: got RGBA(%d,%d,%d,%d)",
			actual.Pix[center], actual.Pix[center+1], actual.Pix[center+2], actual.Pix[center+3])
	}
	for _, off := range []int{0, 8} {
		if actual.Pix[off+3] != 0 {
			t.Errorf("transparent pixel at %d gained alpha: %d", off, actual.Pix[off+3])
		}
	}
}

// Semi-transparent visible pixels are filtered against visible neighbors only
// and keep their original alpha.
func TestBilateralSemiTransparent(t *testing.T) {
	// 3x1: transparent black, 50% gray at 50% alpha, same gray opaque.
	src := newRGBA(3, 1, []uint8{
		0x00, 0x00, 0x00, 0x00,
		0x80, 0x80, 0x80, 0x80,
		0x80, 0x80, 0x80, 0xFF,
	})

	actual := Bilateral(src, 1, 5.0, 100.0)

	mid := 4
	if actual.Pix[mid+3] != 0x80 {
		t.Errorf("alpha must be preserved: got %d, want 128", actual.Pix[mid+3])
	}
	// Both visible neighbors share the same color, so RGB must stay 0x80.
	for c := 0; c < 3; c++ {
		if d := int(actual.Pix[mid+c]) - 0x80; d < -1 || d > 1 {
			t.Errorf("channel %d changed to %d, expected ~128", c, actual.Pix[mid+c])
		}
	}
}

// Exact weighted-average behavior on a hand-computed case.
// 3x3 image, radius 1, spatialSigma 10, colorSigma 300:
//
//	255 255 255
//	  0   0   0
//	  0   0   0
//
// Spatial weights are separable, w(d) = exp(-d^2/200):
//
//	w(0)=1, w(1)=exp(-0.005)=0.995012
//
// Color distance uses all three RGB channels. White vs black differs by 255
// on every channel, so the squared distance is 3*255^2 = 195075 and the
// white color weight is
//
//	exp(-195075/(2*300^2)) = exp(-1.08375) = 0.338337.
//
// Sum of black spatial weights: 1 + 4*0.995012 + 0.995012^2 = 5.97010
// Sum of white spatial weights: 2*0.995012 + 0.995012^2 = 2.98006
//
//	out = 255 * (2.98006 * 0.338337) / (5.97010 + 2.98006 * 0.338337)
//	    = 255 * 1.00826 / 6.97836 = 36.82
//
// which rounds to 37.
func TestBilateralWeightedFormula(t *testing.T) {
	src := newRGBA(3, 3, []uint8{
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF,
		0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF,
	})

	actual := Bilateral(src, 1, 10.0, 300.0)

	center := (1*3 + 1) * 4
	got := int(actual.Pix[center])
	if got < 35 || got > 39 {
		t.Errorf("center pixel = %d, expected ~37 from hand-computed weights", got)
	}
	if actual.Pix[center+3] != 0xFF {
		t.Errorf("alpha changed: %d", actual.Pix[center+3])
	}
}

// Extreme colors (pure black adjacent to pure white) must be handled without
// overflow and produce deterministic, edge-preserving results.
func TestBilateralExtremeColors(t *testing.T) {
	src := newRGBA(4, 4, []uint8{
		0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF,
		0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF,
	})

	// Tiny sigmas: every value must stay an exact black or white pixel, and
	// repeated runs must be byte-identical (deterministic, no overflow drift).
	first := Bilateral(src, 2, 0.1, 0.1)
	second := Bilateral(src, 2, 0.1, 0.1)

	if !util.RGBAImageEqual(first, second) {
		t.Errorf("filter is not deterministic on extreme colors")
	}
	for i := 0; i < len(first.Pix); i += 4 {
		r := first.Pix[i]
		if r != 0x00 && r != 0xFF {
			t.Errorf("extreme black/white edge leaked a mid-tone %d at pixel %d", r, i/4)
		}
		if first.Pix[i+3] != 0xFF {
			t.Errorf("alpha changed at pixel %d: %d", i/4, first.Pix[i+3])
		}
	}
}

// A radius far larger than the image must be clamped instead of allocating a
// gigantic window, and still produce a valid result.
func TestBilateralOversizedRadius(t *testing.T) {
	src := newRGBA(2, 2, []uint8{
		0x10, 0x10, 0x10, 0xFF, 0x20, 0x20, 0x20, 0xFF,
		0x30, 0x30, 0x30, 0xFF, 0x40, 0x40, 0x40, 0xFF,
	})

	actual := Bilateral(src, 1000000, 5.0, 50.0)

	if !actual.Bounds().Eq(src.Bounds()) {
		t.Errorf("bounds changed: %v", actual.Bounds())
	}
	if len(actual.Pix) != len(src.Pix) {
		t.Errorf("output size changed: %d vs %d", len(actual.Pix), len(src.Pix))
	}
}

// Output size and color model rules: always a new *image.RGBA with the source
// bounds, even when the source is a different color model.
func TestBilateralOutputModelAndSize(t *testing.T) {
	src := image.NewGray(image.Rect(0, 0, 3, 2))
	for i := range src.Pix {
		src.Pix[i] = uint8(i * 17)
	}

	actual := Bilateral(src, 1, 2.0, 40.0)

	if _, ok := interface{}(actual).(image.Image).(*image.RGBA); !ok {
		t.Errorf("output must be *image.RGBA, got %T", actual)
	}
	if !actual.Bounds().Eq(src.Bounds()) {
		t.Errorf("bounds changed: got %v, want %v", actual.Bounds(), src.Bounds())
	}
	if actual.Stride != 3*4 {
		t.Errorf("unexpected stride: %d", actual.Stride)
	}
}

// Images whose bounds do not start at the origin must be sampled and written
// with correct offset handling.
func TestBilateralNonZeroBoundsMin(t *testing.T) {
	rect := image.Rect(5, 7, 8, 9)
	src := image.NewRGBA(rect)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 10, G: 20, B: 30, A: 255})
		}
	}

	actual := Bilateral(src, 1, 2.0, 40.0)

	if !actual.Bounds().Eq(rect) {
		t.Fatalf("bounds changed: got %v, want %v", actual.Bounds(), rect)
	}
	// A constant image must survive regardless of its bounds' origin.
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			i := actual.PixOffset(x, y)
			if actual.Pix[i] != 10 || actual.Pix[i+1] != 20 || actual.Pix[i+2] != 30 || actual.Pix[i+3] != 255 {
				t.Errorf("pixel (%d,%d) wrong: %v", x, y, actual.Pix[i:i+4])
			}
		}
	}
}
