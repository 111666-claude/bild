package effect

import (
	"image"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

// cannyGrayImage builds an opaque grayscale RGBA from rows of characters,
// mapping each character through palette to a gray value.
func cannyGrayImage(rows []string, palette map[rune]uint8) *image.RGBA {
	h := len(rows)
	w := len(rows[0])
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y, row := range rows {
		for x, r := range row {
			v := palette[r]
			pos := y*img.Stride + x*4
			img.Pix[pos+0] = v
			img.Pix[pos+1] = v
			img.Pix[pos+2] = v
			img.Pix[pos+3] = 0xFF
		}
	}
	return img
}

// cannyEdgeMap builds an edge map image from rows of characters, where '#'
// is an opaque white edge pixel and any other character is transparent.
func cannyEdgeMap(rows []string) *image.RGBA {
	h := len(rows)
	w := len(rows[0])
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y, row := range rows {
		for x, r := range row {
			if r == '#' {
				pos := y*img.Stride + x*4
				img.Pix[pos+0] = 0xFF
				img.Pix[pos+1] = 0xFF
				img.Pix[pos+2] = 0xFF
				img.Pix[pos+3] = 0xFF
			}
		}
	}
	return img
}

// cannyContentEqual compares two RGBA images by content, ignoring the
// origin of their bounds.
func cannyContentEqual(a, b *image.RGBA) bool {
	if a.Bounds().Dx() != b.Bounds().Dx() || a.Bounds().Dy() != b.Bounds().Dy() {
		return false
	}
	for y := 0; y < a.Bounds().Dy(); y++ {
		for x := 0; x < a.Bounds().Dx(); x++ {
			pa := a.PixOffset(a.Bounds().Min.X+x, a.Bounds().Min.Y+y)
			pb := b.PixOffset(b.Bounds().Min.X+x, b.Bounds().Min.Y+y)
			for c := 0; c < 4; c++ {
				if a.Pix[pa+c] != b.Pix[pb+c] {
					return false
				}
			}
		}
	}
	return true
}

func cannyEdgeCount(img *image.RGBA) int {
	count := 0
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			if img.Pix[y*img.Stride+x*4+3] != 0 {
				count++
			}
		}
	}
	return count
}

func TestCanny(t *testing.T) {
	grays := map[rune]uint8{'0': 0x00, '4': 0x40, '8': 0x80, 'C': 0xC0, 'F': 0xFF}

	cases := []struct {
		description string
		sigma       float64
		low         float64
		high        float64
		value       image.Image
		expected    *image.RGBA
	}{
		{
			description: "empty image",
			sigma:       1,
			low:         20,
			high:        100,
			value:       &image.RGBA{},
			expected:    &image.RGBA{},
		},
		{
			description: "vertical ramp edge",
			sigma:       0,
			low:         20,
			high:        100,
			value: cannyGrayImage([]string{
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
			}, grays),
			expected: cannyEdgeMap([]string{
				"........",
				"...#....",
				"...#....",
				"...#....",
				"...#....",
				"...#....",
				"...#....",
				"........",
			}),
		},
		{
			description: "horizontal ramp edge",
			sigma:       0,
			low:         20,
			high:        100,
			value: cannyGrayImage([]string{
				"00000000",
				"00000000",
				"00000000",
				"88888888",
				"FFFFFFFF",
				"FFFFFFFF",
				"FFFFFFFF",
				"FFFFFFFF",
			}, grays),
			expected: cannyEdgeMap([]string{
				"........",
				"........",
				"........",
				".######.",
				"........",
				"........",
				"........",
				"........",
			}),
		},
		{
			description: "constant image has no edges",
			sigma:       1,
			low:         20,
			high:        100,
			value: cannyGrayImage([]string{
				"8888",
				"8888",
				"8888",
				"8888",
			}, grays),
			expected: cannyEdgeMap([]string{
				"....",
				"....",
				"....",
				"....",
			}),
		},
		{
			description: "constant image with huge sigma has no edges",
			sigma:       math.Inf(1),
			low:         20,
			high:        100,
			value: cannyGrayImage([]string{
				"8888",
				"8888",
				"8888",
				"8888",
			}, grays),
			expected: cannyEdgeMap([]string{
				"....",
				"....",
				"....",
				"....",
			}),
		},
		{
			description: "tiny image has no interior pixels",
			sigma:       0,
			low:         20,
			high:        100,
			value: cannyGrayImage([]string{
				"0F",
				"0F",
			}, grays),
			expected: cannyEdgeMap([]string{
				"..",
				"..",
			}),
		},
		{
			description: "1x1 image",
			sigma:       0,
			low:         20,
			high:        100,
			value: cannyGrayImage([]string{
				"F",
			}, grays),
			expected: cannyEdgeMap([]string{
				".",
			}),
		},
		{
			description: "thresholds above gradient magnitude",
			sigma:       0,
			low:         200,
			high:        250,
			value: cannyGrayImage([]string{
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
				"0008FFFF",
			}, grays),
			expected: cannyEdgeMap([]string{
				"........",
				"........",
				"........",
				"........",
				"........",
				"........",
				"........",
				"........",
			}),
		},
	}

	for _, c := range cases {
		actual := Canny(c.value, c.sigma, c.low, c.high)
		if !util.RGBAImageEqual(actual, c.expected) {
			t.Errorf("%s:\nexpected:\n%v\nactual:\n%v", c.description, util.RGBAToString(c.expected), util.RGBAToString(actual))
		}
	}
}

// TestCannyWeakEdgeBridging verifies that weak edge segments are kept when
// they connect to strong segments, and dropped when they don't. The test
// image is a vertical ramp edge whose contrast decreases smoothly from 240
// (strong) to 80 (weak) along its length; the gradient direction stays
// horizontal everywhere so the edge remains a single connected line.
func TestCannyWeakEdgeBridging(t *testing.T) {
	// Contrast per row: flat strong top, then decreasing by 10 per row.
	vs := []uint8{240, 240, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80}
	h := len(vs)
	w := 8

	build := func(vs []uint8) *image.RGBA {
		img := image.NewRGBA(image.Rect(0, 0, w, h))
		for y, v := range vs {
			for x := 0; x < w; x++ {
				var lum uint8
				switch {
				case x < 3:
					lum = 0
				case x == 3:
					lum = v / 2
				default:
					lum = v
				}
				pos := y*img.Stride + x*4
				img.Pix[pos+0] = lum
				img.Pix[pos+1] = lum
				img.Pix[pos+2] = lum
				img.Pix[pos+3] = 0xFF
			}
		}
		return img
	}

	expected := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 1; y < h-1; y++ {
		pos := y*expected.Stride + 3*4
		expected.Pix[pos+0] = 0xFF
		expected.Pix[pos+1] = 0xFF
		expected.Pix[pos+2] = 0xFF
		expected.Pix[pos+3] = 0xFF
	}

	// The strong top segment seeds the hysteresis and the connected weak
	// tail (rows with contrast below the high threshold) is kept.
	actual := Canny(build(vs), 0, 20, 100)
	if !util.RGBAImageEqual(actual, expected) {
		t.Errorf("bridged weak edge:\nexpected:\n%v\nactual:\n%v", util.RGBAToString(expected), util.RGBAToString(actual))
	}

	// The same edge at uniformly weak contrast has no strong seed, so the
	// entire edge is dropped.
	weak := make([]uint8, h)
	for i := range weak {
		weak[i] = 120
	}
	actual = Canny(build(weak), 0, 20, 100)
	if count := cannyEdgeCount(actual); count != 0 {
		t.Errorf("unconnected weak edge: expected no edges, got %d edge pixels", count)
	}
}

// TestCannySmoothing verifies that the configurable sigma actually smooths:
// an impulse produces edges without smoothing but is suppressed below the
// thresholds with a wide kernel.
func TestCannySmoothing(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 7, 7))
	pos := 3*img.Stride + 3*4
	img.Pix[pos+0] = 0xFF
	img.Pix[pos+1] = 0xFF
	img.Pix[pos+2] = 0xFF
	img.Pix[pos+3] = 0xFF

	if count := cannyEdgeCount(Canny(img, 0, 20, 50)); count == 0 {
		t.Error("expected edges around the impulse without smoothing")
	}
	if count := cannyEdgeCount(Canny(img, 2, 20, 50)); count != 0 {
		t.Errorf("expected impulse to be smoothed away, got %d edge pixels", count)
	}
}

// TestCannyTransparent verifies that fully transparent pixels never create
// edges from their (invisible) color values, and that a real alpha
// discontinuity is detected as an edge.
func TestCannyTransparent(t *testing.T) {
	// Left half opaque black, right half transparent with hidden white.
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			pos := y*img.Stride + x*4
			if x < 4 {
				img.Pix[pos+3] = 0xFF
			} else {
				img.Pix[pos+0] = 0xFF
				img.Pix[pos+1] = 0xFF
				img.Pix[pos+2] = 0xFF
				img.Pix[pos+3] = 0x00
			}
		}
	}

	actual := Canny(img, 0, 20, 100)
	if count := cannyEdgeCount(actual); count != 0 {
		t.Errorf("expected no edges from hidden transparent colors, got %d edge pixels", count)
	}

	// Left half opaque white, right half transparent: the alpha
	// discontinuity is a real edge (a step produces a 2-pixel-wide line).
	for y := 0; y < 8; y++ {
		for x := 0; x < 4; x++ {
			pos := y*img.Stride + x*4
			img.Pix[pos+0] = 0xFF
			img.Pix[pos+1] = 0xFF
			img.Pix[pos+2] = 0xFF
			img.Pix[pos+3] = 0xFF
		}
	}

	expected := cannyEdgeMap([]string{
		"........",
		"...##...",
		"...##...",
		"...##...",
		"...##...",
		"...##...",
		"...##...",
		"........",
	})
	actual = Canny(img, 0, 20, 100)
	if !util.RGBAImageEqual(actual, expected) {
		t.Errorf("alpha edge:\nexpected:\n%v\nactual:\n%v", util.RGBAToString(expected), util.RGBAToString(actual))
	}
}

// TestCannyBounds verifies that images with non-zero bounds produce the same
// edges, offset to the same bounds.
func TestCannyBounds(t *testing.T) {
	grays := map[rune]uint8{'0': 0x00, '8': 0x80, 'F': 0xFF}
	rows := []string{
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
	}

	img := image.NewRGBA(image.Rect(2, 3, 10, 11))
	for y, row := range rows {
		for x, r := range row {
			v := grays[r]
			pos := (y)*img.Stride + (x)*4
			img.Pix[pos+0] = v
			img.Pix[pos+1] = v
			img.Pix[pos+2] = v
			img.Pix[pos+3] = 0xFF
		}
	}

	expected := image.NewRGBA(image.Rect(2, 3, 10, 11))
	for y := 1; y < 7; y++ {
		pos := y*expected.Stride + 3*4
		expected.Pix[pos+0] = 0xFF
		expected.Pix[pos+1] = 0xFF
		expected.Pix[pos+2] = 0xFF
		expected.Pix[pos+3] = 0xFF
	}

	actual := Canny(img, 0, 20, 100)
	if !util.RGBAImageEqual(actual, expected) {
		t.Errorf("non-zero bounds:\nexpected:\n%v\nactual:\n%v", util.RGBAToString(expected), util.RGBAToString(actual))
	}
}

// TestCannyChunkedInputs verifies that the same logical image yields the same
// edges whether it is provided as a contiguous RGBA, a sub-image sharing a
// larger backing array, or an NRGBA.
func TestCannyChunkedInputs(t *testing.T) {
	grays := map[rune]uint8{'0': 0x00, '8': 0x80, 'F': 0xFF}
	rows := []string{
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
	}

	plain := cannyGrayImage(rows, grays)

	// Sub-image of a larger image: non-zero origin and a stride larger than
	// the visible width.
	large := image.NewRGBA(image.Rect(0, 0, 16, 16))
	sub := large.SubImage(image.Rect(4, 5, 12, 13)).(*image.RGBA)
	for y, row := range rows {
		for x, r := range row {
			v := grays[r]
			pos := sub.PixOffset(4+x, 5+y)
			sub.Pix[pos+0] = v
			sub.Pix[pos+1] = v
			sub.Pix[pos+2] = v
			sub.Pix[pos+3] = 0xFF
		}
	}

	nrgba := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y, row := range rows {
		for x, r := range row {
			v := grays[r]
			pos := nrgba.PixOffset(x, y)
			nrgba.Pix[pos+0] = v
			nrgba.Pix[pos+1] = v
			nrgba.Pix[pos+2] = v
			nrgba.Pix[pos+3] = 0xFF
		}
	}

	expected := Canny(plain, 1, 20, 100)

	if actual := Canny(sub, 1, 20, 100); !cannyContentEqual(actual, expected) {
		t.Errorf("sub-image result differs:\nexpected:\n%v\nactual:\n%v", util.RGBAToString(expected), util.RGBAToString(actual))
	}
	if actual := Canny(nrgba, 1, 20, 100); !cannyContentEqual(actual, expected) {
		t.Errorf("NRGBA result differs:\nexpected:\n%v\nactual:\n%v", util.RGBAToString(expected), util.RGBAToString(actual))
	}
}

// TestCannyInvalidParams verifies the deterministic handling of invalid
// parameters by equivalence to their normalized forms.
func TestCannyInvalidParams(t *testing.T) {
	grays := map[rune]uint8{'0': 0x00, '8': 0x80, 'F': 0xFF}
	img := cannyGrayImage([]string{
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
		"0008FFFF",
	}, grays)

	cases := []struct {
		description string
		a           *image.RGBA
		b           *image.RGBA
	}{
		{
			description: "negative sigma disables smoothing",
			a:           Canny(img, -1, 20, 100),
			b:           Canny(img, 0, 20, 100),
		},
		{
			description: "NaN sigma disables smoothing",
			a:           Canny(img, math.NaN(), 20, 100),
			b:           Canny(img, 0, 20, 100),
		},
		{
			description: "swapped thresholds are reordered",
			a:           Canny(img, 0, 100, 20),
			b:           Canny(img, 0, 20, 100),
		},
		{
			description: "NaN low threshold maps to zero",
			a:           Canny(img, 0, math.NaN(), 100),
			b:           Canny(img, 0, 0, 100),
		},
		{
			description: "out of range thresholds are clamped",
			a:           Canny(img, 0, -5, 1e9),
			b:           Canny(img, 0, 0, 255),
		},
		{
			description: "infinite thresholds are clamped",
			a:           Canny(img, 0, math.Inf(-1), math.Inf(1)),
			b:           Canny(img, 0, 0, 255),
		},
		{
			description: "infinite sigma is bounded",
			a:           Canny(img, math.Inf(1), 20, 100),
			b:           Canny(img, math.Inf(1), 20, 100),
		},
	}

	for _, c := range cases {
		if !util.RGBAImageEqual(c.a, c.b) {
			t.Errorf("%s:\nexpected:\n%v\nactual:\n%v", c.description, util.RGBAToString(c.b), util.RGBAToString(c.a))
		}
	}
}

func BenchmarkCanny(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, 512, 512))
	for i := range img.Pix {
		img.Pix[i] = uint8(i % 256)
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		benchResult = Canny(img, 1.4, 50, 100)
	}
}
