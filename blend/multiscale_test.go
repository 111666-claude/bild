package blend

import (
	"image"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

// checkerTexture returns an opaque high-frequency checkerboard texture with a
// slow vertical brightness ramp on top. If invert is true the checkerboard
// phase is flipped.
func checkerTexture(rect image.Rectangle, base, amp uint8, invert bool) *image.RGBA {
	img := image.NewRGBA(rect)
	w := rect.Dx()
	for y := 0; y < rect.Dy(); y++ {
		for x := 0; x < w; x++ {
			v := int(base) + y/4
			phase := (x + y) % 2
			if invert {
				phase = 1 - phase
			}
			if phase == 0 {
				v += int(amp)
			} else {
				v -= int(amp)
			}
			pos := y*img.Stride + x*4
			img.Pix[pos+0] = uint8(v)
			img.Pix[pos+1] = uint8(v)
			img.Pix[pos+2] = uint8(v)
			img.Pix[pos+3] = 0xFF
		}
	}
	return img
}

// solidGray returns an opaque uniformly colored image.
func solidGray(rect image.Rectangle, v uint8) *image.RGBA {
	img := image.NewRGBA(rect)
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = v
		img.Pix[i+1] = v
		img.Pix[i+2] = v
		img.Pix[i+3] = 0xFF
	}
	return img
}

// gradientMask returns a grayscale mask that is black up to x0, white from x1
// on, and a linear ramp in between.
func gradientMask(rect image.Rectangle, x0, x1 int) *image.Gray {
	mask := image.NewGray(rect)
	w := rect.Dx()
	for y := 0; y < rect.Dy(); y++ {
		for x := 0; x < w; x++ {
			var v uint8
			switch {
			case x <= x0:
				v = 0
			case x >= x1:
				v = 0xFF
			default:
				v = uint8((x - x0) * 255 / (x1 - x0))
			}
			mask.Pix[y*mask.Stride+x] = v
		}
	}
	return mask
}

// stepMask returns a grayscale mask that is black on the left half and white
// on the right half.
func stepMask(rect image.Rectangle) *image.Gray {
	mask := image.NewGray(rect)
	w := rect.Dx()
	for y := 0; y < rect.Dy(); y++ {
		for x := 0; x < w; x++ {
			if x >= w/2 {
				mask.Pix[y*mask.Stride+x] = 0xFF
			}
		}
	}
	return mask
}

func maxAdjacentRowDiff(img *image.RGBA, y int) int {
	maxDiff := 0
	pos := y * img.Stride
	for x := 1; x < img.Bounds().Dx(); x++ {
		d := int(img.Pix[pos+x*4]) - int(img.Pix[pos+(x-1)*4])
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

func TestMultiscaleErrors(t *testing.T) {
	img4 := solidGray(image.Rect(0, 0, 4, 4), 0x80)
	img5 := solidGray(image.Rect(0, 0, 5, 4), 0x80)
	mask4 := image.NewGray(image.Rect(0, 0, 4, 4))
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))

	cases := []struct {
		description string
		bg          image.Image
		fg          image.Image
		mask        image.Image
		levels      int
	}{
		{"nil bg", nil, img4, mask4, 3},
		{"nil fg", img4, nil, mask4, 3},
		{"nil mask", img4, img4, nil, 3},
		{"empty bg", empty, img4, mask4, 3},
		{"empty fg", img4, empty, mask4, 3},
		{"size mismatch", img4, img5, mask4, 3},
		{"zero levels", img4, img4, mask4, 0},
		{"negative levels", img4, img4, mask4, -2},
		{"mask too small", img4, img4, image.NewGray(image.Rect(0, 0, 3, 4)), 3},
		{"mask offset away", img4, img4, image.NewGray(image.Rect(10, 10, 14, 14)), 3},
	}

	for _, c := range cases {
		result, err := Multiscale(c.bg, c.fg, c.mask, c.levels)
		if err == nil {
			t.Errorf("%s: expected an error but got none", c.description)
		}
		if result != nil {
			t.Errorf("%s: expected nil result on error", c.description)
		}
	}
}

func TestMultiscaleIdentity(t *testing.T) {
	// Blending an image with itself must reconstruct the input regardless of
	// the mask, the number of levels, or odd dimensions.
	cases := []struct {
		w, h   int
		levels int
	}{
		{32, 32, 1},
		{32, 32, 4},
		{31, 17, 5},
		{17, 31, 3},
		{1, 1, 4},
		{3, 5, 8},
		{64, 48, 32}, // levels beyond the useful depth get clamped
	}

	for _, c := range cases {
		rect := image.Rect(0, 0, c.w, c.h)
		bg := checkerTexture(rect, 100, 30, false)
		fg := checkerTexture(rect, 100, 30, false)
		mask := gradientMask(rect, c.w/4, 3*c.w/4)

		result, err := Multiscale(bg, fg, mask, c.levels)
		if err != nil {
			t.Fatalf("size %dx%d levels %d: unexpected error: %v", c.w, c.h, c.levels, err)
		}
		if !util.RGBAImageApproxEqual(result, bg, 1) {
			t.Errorf("size %dx%d levels %d: expected reconstruction of the input", c.w, c.h, c.levels)
		}
	}
}

func TestMultiscaleConstantRegions(t *testing.T) {
	// With a wide gradient mask, the regions where the mask is saturated must
	// reproduce the corresponding input, high-frequency detail included.
	rect := image.Rect(0, 0, 65, 33)
	bg := checkerTexture(rect, 60, 20, false)
	fg := checkerTexture(rect, 180, 20, true)
	mask := gradientMask(rect, 8, 56)

	result, err := Multiscale(bg, fg, mask, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Deep inside the mask == 0 region the result must equal bg, and deep
	// inside the mask == 1 region it must equal fg.
	for y := 0; y < rect.Dy(); y++ {
		for _, x := range []int{0, 1, 2, 62, 63, 64} {
			pos := y*result.Stride + x*4
			var want *image.RGBA
			if x <= 2 {
				want = bg
			} else {
				want = fg
			}
			wpos := y*want.Stride + x*4
			for c := 0; c < 4; c++ {
				d := int(result.Pix[pos+c]) - int(want.Pix[wpos+c])
				if d < -2 || d > 2 {
					t.Fatalf("pixel (%d,%d) channel %d: got %d, want %d +/- 2",
						x, y, c, result.Pix[pos+c], want.Pix[wpos+c])
				}
			}
		}
	}
}

func TestMultiscaleSeamSmoothness(t *testing.T) {
	// A hard step mask between two constant regions must reconstruct as a
	// smooth transition instead of a visible jump.
	rect := image.Rect(0, 0, 64, 32)
	bg := solidGray(rect, 40)
	fg := solidGray(rect, 200)
	mask := stepMask(rect)

	result, err := Multiscale(bg, fg, mask, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	const step = 200 - 40
	for y := 0; y < rect.Dy(); y++ {
		if d := maxAdjacentRowDiff(result, y); d > step/4 {
			t.Errorf("row %d: adjacent pixel jump of %d is a visible seam (input step %d)", y, d, step)
		}
	}

	// The transition must be monotonic across the seam.
	y := rect.Dy() / 2
	pos := y * result.Stride
	for x := 1; x < rect.Dx(); x++ {
		if result.Pix[pos+x*4] < result.Pix[pos+(x-1)*4] {
			t.Fatalf("row %d: non-monotonic transition at x=%d (%d -> %d)",
				y, x, result.Pix[pos+(x-1)*4], result.Pix[pos+x*4])
		}
	}

	// The extremes must still reach the input values.
	if v := result.Pix[pos]; v > 40+8 {
		t.Errorf("left edge: got %d, want close to 40", v)
	}
	if v := result.Pix[pos+(rect.Dx()-1)*4]; v < 200-8 {
		t.Errorf("right edge: got %d, want close to 200", v)
	}
}

func TestMultiscaleTextureAcrossGradient(t *testing.T) {
	// High-frequency textures stitched with a wide gradient mask: the column
	// mean luminance must move smoothly and monotonically from the bg level to
	// the fg level without a seam or overshoot, and must stay within the range
	// of the two inputs.
	rect := image.Rect(0, 0, 96, 48)
	bg := checkerTexture(rect, 60, 20, false)
	fg := checkerTexture(rect, 180, 20, true)
	mask := gradientMask(rect, 16, 80)

	result, err := Multiscale(bg, fg, mask, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	h := rect.Dy()
	w := rect.Dx()
	prevMean := -1.0
	for x := 0; x < w; x++ {
		sum := 0
		for y := 0; y < h; y++ {
			sum += int(result.Pix[y*result.Stride+x*4])
		}
		mean := float64(sum) / float64(h)

		if mean < 60-6 || mean > 180+6 {
			t.Fatalf("column %d: mean %.1f overshoots the input range [60, 180]", x, mean)
		}
		if prevMean >= 0 && mean < prevMean-0.5 {
			t.Fatalf("column %d: mean %.1f drops below previous %.1f, seam is not monotonic", x, mean, prevMean)
		}
		if prevMean >= 0 && mean-prevMean > 8 {
			t.Fatalf("column %d: mean jumps by %.1f, seam is not smooth", x, mean-prevMean)
		}
		prevMean = mean
	}
}

func TestMultiscaleNonZeroBoundsAndImmutability(t *testing.T) {
	bgRect := image.Rect(3, 2, 34, 19)
	fgRect := image.Rect(-7, 11, 24, 28) // same size, different origin
	bg := checkerTexture(bgRect, 70, 25, false)
	fg := checkerTexture(fgRect, 170, 25, true)
	mask := gradientMask(bgRect, 8, 24)

	bgCopy := append([]uint8(nil), bg.Pix...)
	fgCopy := append([]uint8(nil), fg.Pix...)
	maskCopy := append([]uint8(nil), mask.Pix...)

	result, err := Multiscale(bg, fg, mask, 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Bounds().Dx() != bgRect.Dx() || result.Bounds().Dy() != bgRect.Dy() {
		t.Errorf("result bounds %v do not match input size %dx%d", result.Bounds(), bgRect.Dx(), bgRect.Dy())
	}
	if result.Bounds() != bgRect {
		t.Errorf("result bounds %v, want %v", result.Bounds(), bgRect)
	}

	if !equalBytes(bg.Pix, bgCopy) {
		t.Errorf("bg was modified")
	}
	if !equalBytes(fg.Pix, fgCopy) {
		t.Errorf("fg was modified")
	}
	if !equalBytes(mask.Pix, maskCopy) {
		t.Errorf("mask was modified")
	}
}

func TestMultiscaleTransparentEdges(t *testing.T) {
	// A fully transparent region must contribute no color and blend its alpha
	// like any other channel.
	rect := image.Rect(0, 0, 16, 16)
	bg := image.NewRGBA(rect) // fully transparent
	fg := solidGray(rect, 0xFF)
	mask := image.NewGray(rect)
	for i := range mask.Pix {
		mask.Pix[i] = 0xFF // fully select fg
	}

	result, err := Multiscale(bg, fg, mask, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !util.RGBAImageApproxEqual(result, fg, 2) {
		t.Errorf("mask fully selecting fg: expected fg, got %v", result.Pix[:8])
	}

	// Half-selected transparent input: alpha must land halfway and the
	// visible color must come from the opaque source only.
	for i := range mask.Pix {
		mask.Pix[i] = 0x80
	}
	result, err = Multiscale(bg, fg, mask, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pos := 8*result.Stride + 8*4
	a := int(result.Pix[pos+3])
	if a < 120 || a > 136 {
		t.Errorf("alpha: got %d, want about 128", a)
	}
	r := int(result.Pix[pos+0])
	if r < 240 {
		// Premultiplied fusion keeps the opaque source's color instead of
		// diluting it with the transparent black of bg.
		t.Errorf("color leaked from transparent region: got R=%d, want close to 255", r)
	}
}

func TestMultiscaleGrayMaskSource(t *testing.T) {
	// Mask weights must come from luminance so colored mask inputs work too.
	rect := image.Rect(0, 0, 8, 8)
	bg := solidGray(rect, 0)
	fg := solidGray(rect, 0xFF)
	mask := image.NewRGBA(rect)
	for i := 0; i < len(mask.Pix); i += 4 {
		mask.Pix[i+0] = 0xFF // pure red has luminance ~0.299
		mask.Pix[i+3] = 0xFF
	}

	result, err := Multiscale(bg, fg, mask, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	v := int(result.Pix[0])
	if v < 60 || v > 95 {
		t.Errorf("got %d, want roughly 0.299*255 ~= 76", v)
	}
}

func equalBytes(a, b []uint8) bool {
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
