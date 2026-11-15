package segment

import (
	"fmt"
	"image"
	"image/color"
	"strings"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

func TestSDFRectangle(t *testing.T) {
	// 7x7 image with an opaque 3x3 rectangle at the center.
	img := image.NewRGBA(image.Rect(0, 0, 7, 7))
	for y := 2; y <= 4; y++ {
		for x := 2; x <= 4; x++ {
			img.SetRGBA(x, y, imageRGBAWhite)
		}
	}

	expected := &image.Gray{
		Rect:   image.Rect(0, 0, 7, 7),
		Stride: 7,
		Pix: []uint8{
			0, 0, 0, 0, 0, 0, 0,
			0, 37, 64, 64, 64, 37, 0,
			0, 64, 191, 191, 191, 64, 0,
			0, 64, 191, 255, 191, 64, 0,
			0, 64, 191, 191, 191, 64, 0,
			0, 37, 64, 64, 64, 37, 0,
			0, 0, 0, 0, 0, 0, 0,
		},
	}

	actual := SDF(img, 128, 2)
	if !util.GrayImageEqual(actual, expected) {
		t.Errorf("SDF rectangle:\nexpected:\n%v\nactual:\n%v", grayToString(expected), grayToString(actual))
	}
}

func TestSDFRectangleNonZeroBounds(t *testing.T) {
	// Same shape as TestSDFRectangle but with a non-zero origin. The output
	// bounds must match the input bounds and the pixel data must be identical
	// to the zero-origin case.
	img := image.NewRGBA(image.Rect(10, 20, 17, 27))
	for y := 22; y <= 24; y++ {
		for x := 12; x <= 14; x++ {
			img.SetRGBA(x, y, imageRGBAWhite)
		}
	}

	expected := &image.Gray{
		Rect:   image.Rect(10, 20, 17, 27),
		Stride: 7,
		Pix: []uint8{
			0, 0, 0, 0, 0, 0, 0,
			0, 37, 64, 64, 64, 37, 0,
			0, 64, 191, 191, 191, 64, 0,
			0, 64, 191, 255, 191, 64, 0,
			0, 64, 191, 191, 191, 64, 0,
			0, 37, 64, 64, 64, 37, 0,
			0, 0, 0, 0, 0, 0, 0,
		},
	}

	actual := SDF(img, 128, 2)
	if actual.Bounds() != img.Bounds() {
		t.Fatalf("SDF bounds: expected %v got %v", img.Bounds(), actual.Bounds())
	}
	if !util.GrayImageEqual(actual, expected) {
		t.Errorf("SDF non-zero bounds:\nexpected:\n%v\nactual:\n%v", grayToString(expected), grayToString(actual))
	}
}

func TestSDFCircle(t *testing.T) {
	// 9x9 image with a filled circle of radius 3 centered at (4,4).
	img := image.NewRGBA(image.Rect(0, 0, 9, 9))
	for y := 0; y < 9; y++ {
		for x := 0; x < 9; x++ {
			dx, dy := x-4, y-4
			if dx*dx+dy*dy <= 9 {
				img.SetRGBA(x, y, imageRGBAWhite)
			}
		}
	}

	actual := SDF(img, 128, 3)
	w, h := 9, 9
	at := func(x, y int) int {
		return int(actual.Pix[y*actual.Stride+x])
	}

	// The center is as far inside as the spread reaches.
	if at(4, 4) != 255 {
		t.Errorf("expected center 255, got %d", at(4, 4))
	}
	// Corners are at distance sqrt(8) from the nearest circle pixel (2,2),
	// i.e. 127.5 - 127.5*sqrt(8)/3 = 7.
	for _, p := range [][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
		if at(p[0], p[1]) != 7 {
			t.Errorf("expected corner %v to be 7, got %d", p, at(p[0], p[1]))
		}
	}
	// Every inside pixel must be above the contour value and every outside
	// pixel below it.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := x-4, y-4
			inside := dx*dx+dy*dy <= 9
			v := at(x, y)
			if inside && v <= 128 {
				t.Errorf("inside pixel (%d,%d): expected > 128, got %d", x, y, v)
			}
			if !inside && v >= 128 {
				t.Errorf("outside pixel (%d,%d): expected < 128, got %d", x, y, v)
			}
		}
	}
	// Values must increase monotonically along a row towards the center.
	for x := 0; x < 4; x++ {
		if at(x, 4) > at(x+1, 4) {
			t.Errorf("row not monotonic towards center: (%d,4)=%d > (%d,4)=%d", x, at(x, 4), x+1, at(x+1, 4))
		}
	}
}

func TestSDFHole(t *testing.T) {
	// 7x7 fully opaque image with a single transparent pixel hole at (3,3).
	img := image.NewRGBA(image.Rect(0, 0, 7, 7))
	for y := 0; y < 7; y++ {
		for x := 0; x < 7; x++ {
			img.SetRGBA(x, y, imageRGBAWhite)
		}
	}
	img.SetRGBA(3, 3, color.RGBA{})

	expected := &image.Gray{
		Rect:   image.Rect(0, 0, 7, 7),
		Stride: 7,
		Pix: []uint8{
			255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 255, 255,
			255, 255, 218, 191, 218, 255, 255,
			255, 255, 191, 64, 191, 255, 255,
			255, 255, 218, 191, 218, 255, 255,
			255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 255, 255,
		},
	}

	actual := SDF(img, 128, 2)
	if !util.GrayImageEqual(actual, expected) {
		t.Errorf("SDF hole:\nexpected:\n%v\nactual:\n%v", grayToString(expected), grayToString(actual))
	}
}

func TestSDFSemiTransparentEdge(t *testing.T) {
	// 3x1 row: transparent, semi-transparent, opaque.
	img := &image.RGBA{
		Rect:   image.Rect(0, 0, 3, 1),
		Stride: 3 * 4,
		Pix: []uint8{
			0xFF, 0xFF, 0xFF, 0x00,
			0xFF, 0xFF, 0xFF, 0xC8, // alpha 200
			0xFF, 0xFF, 0xFF, 0xFF,
		},
	}

	// The semi-transparent pixel lands between the contour and the fully
	// opaque value, at a sub-pixel offset proportional to its coverage:
	// (200-128)/127 * 127.5 + 127.5 = 200.
	expected := &image.Gray{
		Rect:   image.Rect(0, 0, 3, 1),
		Stride: 3,
		Pix:    []uint8{0, 200, 255},
	}

	actual := SDF(img, 128, 1)
	if !util.GrayImageEqual(actual, expected) {
		t.Errorf("SDF semi-transparent:\nexpected:\n%v\nactual:\n%v", grayToString(expected), grayToString(actual))
	}
}

func TestSDFUniform(t *testing.T) {
	cases := []struct {
		desc     string
		alpha    uint8
		expected uint8
	}{
		{"fully transparent", 0x00, 0x00},
		{"below threshold", 0x40, 0x00},
		{"fully opaque", 0xFF, 0xFF},
		{"above threshold", 0xC0, 0xFF},
	}

	for _, c := range cases {
		img := image.NewRGBA(image.Rect(0, 0, 8, 8))
		for i := 3; i < len(img.Pix); i += 4 {
			img.Pix[i] = c.alpha
		}

		actual := SDF(img, 128, 4)
		for i, v := range actual.Pix {
			if v != c.expected {
				t.Errorf("%s: expected uniform %d, got %d at pixel %d", c.desc, c.expected, v, i)
				break
			}
		}
	}
}

func TestSDFThresholdBoundary(t *testing.T) {
	// Alpha exactly at the threshold is inside the shape.
	inside := &image.RGBA{
		Rect:   image.Rect(0, 0, 1, 1),
		Stride: 4,
		Pix:    []uint8{0xFF, 0xFF, 0xFF, 128},
	}
	if v := SDF(inside, 128, 1).Pix[0]; v != 0xFF {
		t.Errorf("alpha == threshold: expected inside (255), got %d", v)
	}

	outside := &image.RGBA{
		Rect:   image.Rect(0, 0, 1, 1),
		Stride: 4,
		Pix:    []uint8{0xFF, 0xFF, 0xFF, 127},
	}
	if v := SDF(outside, 128, 1).Pix[0]; v != 0x00 {
		t.Errorf("alpha < threshold: expected outside (0), got %d", v)
	}
}

func TestSDFInvalidSpread(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 7, 7))
	for y := 2; y <= 4; y++ {
		for x := 2; x <= 4; x++ {
			img.SetRGBA(x, y, imageRGBAWhite)
		}
	}

	reference := SDF(img, 128, 1)
	for _, spread := range []float64{0, -1, -100} {
		actual := SDF(img, 128, spread)
		if !util.GrayImageEqual(actual, reference) {
			t.Errorf("spread %v: expected result identical to spread 1", spread)
		}
	}
}

func TestSDFLargeImage(t *testing.T) {
	// Exercises the distance transform on a larger image; a quadratic
	// per-pixel search would make this test noticeably slow.
	const size = 1024
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := size / 4; y < 3*size/4; y++ {
		for x := size / 4; x < 3*size/4; x++ {
			img.SetRGBA(x, y, imageRGBAWhite)
		}
	}

	actual := SDF(img, 128, 8)
	if v := actual.Pix[(size/2)*actual.Stride+size/2]; v != 255 {
		t.Errorf("expected center 255, got %d", v)
	}
	if v := actual.Pix[0]; v != 0 {
		t.Errorf("expected corner 0, got %d", v)
	}
}

var imageRGBAWhite = color.RGBA{0xFF, 0xFF, 0xFF, 0xFF}

func grayToString(img *image.Gray) string {
	var sb strings.Builder
	b := img.Bounds()
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			fmt.Fprintf(&sb, "%3d ", img.Pix[y*img.Stride+x])
		}
		sb.WriteString("\n")
	}
	return sb.String()
}
