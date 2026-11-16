package guidedfilter

import (
	"image"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

// noisyStep returns an 8x1 grayscale step image with added noise along with
// a clean 8x1 step guide. The left half is dark, the right half bright.
func noisyStep() (src *image.RGBA, guide *image.Gray) {
	vals := []uint8{10, 0, 20, 10, 245, 255, 235, 255}
	src = image.NewRGBA(image.Rect(0, 0, 8, 1))
	guide = image.NewGray(image.Rect(0, 0, 8, 1))
	for x, v := range vals {
		src.Pix[x*4+0] = v
		src.Pix[x*4+1] = v
		src.Pix[x*4+2] = v
		src.Pix[x*4+3] = 0xFF
		if x < 4 {
			guide.Pix[x] = 0
		} else {
			guide.Pix[x] = 255
		}
	}
	return src, guide
}

func TestFilterNoisyStep(t *testing.T) {
	src, guide := noisyStep()
	srcCopy := image.NewRGBA(src.Bounds())
	copy(srcCopy.Pix, src.Pix)
	guideCopy := image.NewGray(guide.Bounds())
	copy(guideCopy.Pix, guide.Pix)

	actual := Filter(src, guide, 1, 0.04)

	// Hand-computed from the guided filter definition: the step edge stays
	// sharp (24 vs 235 across the boundary) while the noisy flat regions
	// are smoothed towards their local means.
	expected := &image.RGBA{
		Rect:   image.Rect(0, 0, 8, 1),
		Stride: 8 * 4,
		Pix: []uint8{
			0x08, 0x08, 0x08, 0xFF, 0x08, 0x08, 0x08, 0xFF,
			0x10, 0x10, 0x10, 0xFF, 0x18, 0x18, 0x18, 0xFF,
			0xEB, 0xEB, 0xEB, 0xFF, 0xF4, 0xF4, 0xF4, 0xFF,
			0xF6, 0xF6, 0xF6, 0xFF, 0xF7, 0xF7, 0xF7, 0xFF,
		},
	}

	if !util.RGBAImageApproxEqual(actual, expected, 1) {
		t.Errorf("expected: %v actual: %v", expected.Pix, actual.Pix)
	}

	// Inputs must not be mutated.
	if !util.RGBAImageEqual(src, srcCopy) {
		t.Error("source image was mutated")
	}
	for i, v := range guide.Pix {
		if v != guideCopy.Pix[i] {
			t.Error("guide image was mutated")
			break
		}
	}
}

func TestFilterColorSource(t *testing.T) {
	// Constant channels must pass through unchanged regardless of the guide,
	// while the noisy channel follows the guide's edges.
	src, guide := noisyStep()
	for x := 0; x < 8; x++ {
		src.Pix[x*4+1] = 100
		src.Pix[x*4+2] = 200
	}

	actual := Filter(src, guide, 1, 0.04)

	expectedR := []uint8{0x08, 0x08, 0x10, 0x18, 0xEB, 0xF4, 0xF6, 0xF7}
	for x := 0; x < 8; x++ {
		r := actual.Pix[x*4+0]
		g := actual.Pix[x*4+1]
		b := actual.Pix[x*4+2]
		if d := int(r) - int(expectedR[x]); d < -1 || d > 1 {
			t.Errorf("x=%d: expected R~%d actual: %d", x, expectedR[x], r)
		}
		if g != 100 {
			t.Errorf("x=%d: expected G=100 actual: %d", x, g)
		}
		if b != 200 {
			t.Errorf("x=%d: expected B=200 actual: %d", x, b)
		}
	}
}

func TestFilterZeroVarianceGuide(t *testing.T) {
	// A flat guide has zero local variance; with epsilon 0 the filter must
	// fall back to local averaging without producing NaN.
	src := &image.RGBA{
		Rect:   image.Rect(0, 0, 3, 1),
		Stride: 3 * 4,
		Pix: []uint8{
			0x00, 0x00, 0x00, 0xFF, 0x80, 0x80, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		},
	}
	guide := image.NewGray(image.Rect(0, 0, 3, 1))
	for i := range guide.Pix {
		guide.Pix[i] = 128
	}

	actual := Filter(src, guide, 1, 0)

	expected := &image.RGBA{
		Rect:   image.Rect(0, 0, 3, 1),
		Stride: 3 * 4,
		Pix: []uint8{
			0x60, 0x60, 0x60, 0xFF, 0x80, 0x80, 0x80, 0xFF, 0xA0, 0xA0, 0xA0, 0xFF,
		},
	}

	if !util.RGBAImageApproxEqual(actual, expected, 1) {
		t.Errorf("expected: %v actual: %v", expected.Pix, actual.Pix)
	}
	for i, v := range actual.Pix {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN at pix %d", i)
		}
	}
}

func TestFilterNonFiniteEpsilon(t *testing.T) {
	src := &image.RGBA{
		Rect:   image.Rect(0, 0, 3, 1),
		Stride: 3 * 4,
		Pix: []uint8{
			0x00, 0x00, 0x00, 0xFF, 0x80, 0x80, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
		},
	}
	guide := image.NewGray(image.Rect(0, 0, 3, 1))
	for i := range guide.Pix {
		guide.Pix[i] = 128
	}

	expected := Filter(src, guide, 1, 0)

	for _, eps := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1} {
		actual := Filter(src, guide, 1, eps)
		if !util.RGBAImageEqual(actual, expected) {
			t.Errorf("epsilon=%v: expected: %v actual: %v", eps, expected.Pix, actual.Pix)
		}
	}
}

func TestFilterInvalidParams(t *testing.T) {
	src, guide := noisyStep()

	cases := []struct {
		name   string
		src    image.Image
		guide  image.Image
		radius int
	}{
		{name: "zero radius", src: src, guide: guide, radius: 0},
		{name: "negative radius", src: src, guide: guide, radius: -3},
		{name: "mismatched guide size", src: src, guide: image.NewGray(image.Rect(0, 0, 4, 1)), radius: 1},
		{name: "mismatched guide height", src: src, guide: image.NewGray(image.Rect(0, 0, 8, 2)), radius: 1},
	}

	for _, c := range cases {
		actual := Filter(c.src, c.guide, c.radius, 0.04)
		if !util.RGBAImageEqual(actual, src) {
			t.Errorf("%s: expected a copy of the source", c.name)
		}
	}
}

func TestFilterEmptyImage(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 0, 0))
	guide := image.NewGray(image.Rect(0, 0, 0, 0))

	actual := Filter(src, guide, 1, 0.04)
	if actual.Bounds().Dx() != 0 || actual.Bounds().Dy() != 0 {
		t.Errorf("expected empty result, got %v", actual.Bounds())
	}
}

func TestFilterNonZeroBounds(t *testing.T) {
	vals := []uint8{10, 0, 20, 10, 245, 255, 235, 255}
	src := image.NewRGBA(image.Rect(2, 3, 10, 4))
	guide := image.NewGray(image.Rect(5, 7, 13, 8))
	for x, v := range vals {
		pos := x * 4
		src.Pix[pos+0] = v
		src.Pix[pos+1] = v
		src.Pix[pos+2] = v
		src.Pix[pos+3] = 0xFF
		if x < 4 {
			guide.Pix[x] = 0
		} else {
			guide.Pix[x] = 255
		}
	}

	actual := Filter(src, guide, 1, 0.04)

	if actual.Bounds() != src.Bounds() {
		t.Fatalf("expected bounds %v actual: %v", src.Bounds(), actual.Bounds())
	}

	expected := []uint8{0x08, 0x08, 0x10, 0x18, 0xEB, 0xF4, 0xF6, 0xF7}
	for x, e := range expected {
		if v := actual.Pix[x*4]; v < e-1 || v > e+1 {
			t.Errorf("x=%d: expected ~%d actual: %d", x, e, v)
		}
	}
}

func TestFilterAlphaPreserved(t *testing.T) {
	src, guide := noisyStep()
	src.Pix[0*4+3] = 0x80
	src.Pix[5*4+3] = 0x00

	actual := Filter(src, guide, 1, 0.04)

	if actual.Pix[0*4+3] != 0x80 {
		t.Errorf("expected alpha 0x80 actual: %#x", actual.Pix[0*4+3])
	}
	if actual.Pix[5*4+3] != 0x00 {
		t.Errorf("expected alpha 0x00 actual: %#x", actual.Pix[5*4+3])
	}
}
