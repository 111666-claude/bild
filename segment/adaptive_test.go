package segment

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/util"
)

// grayAt returns the perception-ranked luminance used by the segment package.
func grayAt(c color.RGBA) float64 {
	return util.Rank(c)
}

func isTransparent(c []uint8) bool {
	return c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0
}

// naiveSauvola is a deliberately simple, independent reimplementation of the
// Sauvola decision: it recomputes every clipped neighborhood from scratch.
// Tests use it as an oracle for the summed-area-table implementation.
func naiveSauvola(img *image.RGBA, window int, k, r float64) *image.Gray {
	bounds := img.Bounds()
	dst := image.NewGray(bounds)
	radius := window / 2

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pos := img.PixOffset(x, y)
			c := img.Pix[pos : pos+4]
			outPos := dst.PixOffset(x, y)
			if isTransparent(c) {
				dst.Pix[outPos] = 0xFF
				continue
			}

			top := y - radius
			if top < bounds.Min.Y {
				top = bounds.Min.Y
			}
			bottom := y + radius
			if bottom > bounds.Max.Y-1 {
				bottom = bounds.Max.Y - 1
			}
			left := x - radius
			if left < bounds.Min.X {
				left = bounds.Min.X
			}
			right := x + radius
			if right > bounds.Max.X-1 {
				right = bounds.Max.X - 1
			}

			var n, sum, sqSum float64
			for ny := top; ny <= bottom; ny++ {
				for nx := left; nx <= right; nx++ {
					p := img.PixOffset(nx, ny)
					nc := img.Pix[p : p+4]
					if isTransparent(nc) {
						continue
					}
					lum := grayAt(color.RGBA{nc[0], nc[1], nc[2], nc[3]})
					n++
					sum += lum
					sqSum += lum * lum
				}
			}

			mean := sum / n
			variance := sqSum/n - mean*mean
			if variance < 0 {
				variance = 0
			}
			threshold := mean * (1 + k*(math.Sqrt(variance)/r-1))

			lum := grayAt(color.RGBA{c[0], c[1], c[2], c[3]})
			if lum < threshold {
				dst.Pix[outPos] = 0x00
			} else {
				dst.Pix[outPos] = 0xFF
			}
		}
	}
	return dst
}

// gradientLineArt builds a synthetic scan of line art under uneven
// illumination: the background luminance ramps horizontally from a dark
// shadow on the left to an overexposed right side, while two vertical dark
// strokes run through the whole height.
func gradientLineArt(width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	inkColumns := map[int]bool{2: true, width - 4: true}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			lum := uint8(70 + x*10)
			if inkColumns[x] {
				lum = 30
			}
			pos := img.PixOffset(x, y)
			img.Pix[pos+0] = lum
			img.Pix[pos+1] = lum
			img.Pix[pos+2] = lum
			img.Pix[pos+3] = 0xFF
		}
	}
	return img
}

func TestSauvolaThresholdGradientLineArt(t *testing.T) {
	const width, height = 15, 7
	src := gradientLineArt(width, height)

	result, err := SauvolaThreshold(src, 5, DefaultSauvolaK, DefaultSauvolaR)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Bounds().Eq(src.Bounds()) {
		t.Fatalf("bounds mismatch: got %v want %v", result.Bounds(), src.Bounds())
	}

	inkColumns := map[int]bool{2: true, width - 4: true}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			got := result.GrayAt(x, y).Y
			switch {
			case inkColumns[x]:
				// Every stroke pixel must survive on the shadowed left side
				// and the bright right side, including clipped edge windows.
				if got != 0x00 {
					t.Errorf("ink at (%d,%d) lost: got %#x want black", x, y, got)
				}
			case x == 0 || x == 7 || x == width-1:
				// Background far from any stroke stays white on both sides of
				// the illumination gradient, including border pixels.
				if got != 0xFF {
					t.Errorf("background at (%d,%d) swallowed: got %#x want white", x, y, got)
				}
			}
		}
	}

	// Edge rows must decide identically to the middle row because the
	// pattern is uniform along Y and edge windows are clipped, not extended.
	for x := 0; x < width; x++ {
		top := result.GrayAt(x, 0).Y
		middle := result.GrayAt(x, height/2).Y
		bottom := result.GrayAt(x, height-1).Y
		if top != middle || bottom != middle {
			t.Errorf("edge row mismatch at x=%d: top %#x middle %#x bottom %#x", x, top, middle, bottom)
		}
	}

	// A single global level cannot cope with the gradient: at level 128 the
	// shadowed background on the left is swallowed as foreground, while the
	// adaptive result keeps it background.
	global := Threshold(src, 128)
	if global.GrayAt(0, 0).Y != 0x00 {
		t.Fatal("test setup changed: expected global threshold to fail on the dark side")
	}
	if result.GrayAt(0, 0).Y != 0xFF {
		t.Errorf("adaptive threshold should keep the dark-side background white")
	}
}

func TestSauvolaThresholdMatchesNeighborhoodComputation(t *testing.T) {
	// Non-zero origin, with a transparent pixel mixed in.
	src := image.NewRGBA(image.Rect(3, 4, 8, 9))
	values := [5][5]uint8{
		{10, 200, 90, 150, 40},
		{250, 30, 180, 60, 220},
		{100, 140, 0, 210, 75},
		{55, 235, 120, 25, 190},
		{170, 85, 205, 130, 45},
	}
	for j := 0; j < 5; j++ {
		for i := 0; i < 5; i++ {
			pos := src.PixOffset(3+i, 4+j)
			if i == 2 && j == 2 {
				// Fully transparent pixel.
				src.Pix[pos+3] = 0x00
				continue
			}
			v := values[j][i]
			src.Pix[pos+0] = v
			src.Pix[pos+1] = v
			src.Pix[pos+2] = v
			src.Pix[pos+3] = 0xFF
		}
	}

	for _, window := range []int{3, 5} {
		result, err := SauvolaThreshold(src, window, 0.24, 128)
		if err != nil {
			t.Fatalf("window %d: unexpected error: %v", window, err)
		}
		expected := naiveSauvola(src, window, 0.24, 128)
		if !result.Bounds().Eq(src.Bounds()) {
			t.Fatalf("window %d: bounds got %v want %v", window, result.Bounds(), src.Bounds())
		}
		if !util.GrayImageEqual(result, expected) {
			t.Fatalf("window %d: result %v does not match direct neighborhood result %v", window, result, expected)
		}
	}

	// The transparent pixel itself is always background.
	if result, _ := SauvolaThreshold(src, 3, 0.2, 128); result.GrayAt(5, 6).Y != 0xFF {
		t.Errorf("transparent pixel must be white, got %#x", result.GrayAt(5, 6).Y)
	}
}

func TestSauvolaThresholdTransparentExcludedFromStats(t *testing.T) {
	// Two identical images except one carries a transparent pixel where the
	// other has an opaque white pixel. Neighboring decisions must differ,
	// proving transparent pixels do not enter the neighborhood statistics,
	// while the transparent output pixel itself stays white.
	build := func(transparent bool) *image.RGBA {
		img := image.NewRGBA(image.Rect(0, 0, 3, 3))
		for y := 0; y < 3; y++ {
			for x := 0; x < 3; x++ {
				pos := img.PixOffset(x, y)
				v := uint8(100)
				img.Pix[pos+0], img.Pix[pos+1], img.Pix[pos+2] = v, v, v
				img.Pix[pos+3] = 0xFF
			}
		}
		if transparent {
			pos := img.PixOffset(0, 0)
			img.Pix[pos+0], img.Pix[pos+1], img.Pix[pos+2], img.Pix[pos+3] = 0, 0, 0, 0
		} else {
			pos := img.PixOffset(0, 0)
			img.Pix[pos+0], img.Pix[pos+1], img.Pix[pos+2] = 0xFF, 0xFF, 0xFF
		}
		return img
	}

	withTransparent, err := SauvolaThreshold(build(true), 3, 0.2, 128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	withWhite, err := SauvolaThreshold(build(false), 3, 0.2, 128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if withTransparent.GrayAt(0, 0).Y != 0xFF {
		t.Errorf("transparent output pixel must be white, got %#x", withTransparent.GrayAt(0, 0).Y)
	}
	if withTransparent.GrayAt(1, 1).Y == withWhite.GrayAt(1, 1).Y {
		t.Error("transparent pixel must be excluded from neighborhood statistics")
	}
}

func TestSauvolaThresholdErrors(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 5, 5))

	cases := []struct {
		name   string
		img    image.Image
		window int
		k      float64
		r      float64
		err    error
	}{
		{"zero window", src, 0, 0.2, 128, errOddWindow},
		{"negative window", src, -3, 0.2, 128, errOddWindow},
		{"even window", src, 4, 0.2, 128, errOddWindow},
		{"NaN k", src, 3, math.NaN(), 128, errNonFiniteK},
		{"positive infinite k", src, 3, math.Inf(1), 128, errNonFiniteK},
		{"negative infinite k", src, 3, math.Inf(-1), 128, errNonFiniteK},
		{"NaN r", src, 3, 0.2, math.NaN(), errNonPositiveR},
		{"infinite r", src, 3, 0.2, math.Inf(1), errNonPositiveR},
		{"zero r", src, 3, 0.2, 0, errNonPositiveR},
		{"negative r", src, 3, 0.2, -1, errNonPositiveR},
		{"window larger than image", src, 7, 0.2, 128, errWindowTooLarge},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result, err := SauvolaThreshold(c.img, c.window, c.k, c.r)
			if err != c.err {
				t.Fatalf("got error %v want %v", err, c.err)
			}
			if result != nil {
				t.Fatalf("expected nil image on error, got %v", result)
			}
		})
	}

	// Window larger than only one dimension is still rejected.
	narrow := image.NewRGBA(image.Rect(0, 0, 3, 5))
	if _, err := SauvolaThreshold(narrow, 5, 0.2, 128); err != errWindowTooLarge {
		t.Fatalf("got error %v want %v", err, errWindowTooLarge)
	}

	// Window equal to both dimensions is the largest valid value.
	if _, err := SauvolaThreshold(src, 5, 0.2, 128); err != nil {
		t.Fatalf("window equal to image dimensions should be accepted, got: %v", err)
	}
}

func TestSauvolaThresholdEmptyImage(t *testing.T) {
	empty := image.NewRGBA(image.Rect(0, 0, 0, 0))
	result, err := SauvolaThreshold(empty, 3, 0.2, 128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || !result.Bounds().Eq(empty.Bounds()) || len(result.Pix) != 0 {
		t.Fatalf("expected empty Gray with bounds %v, got %v", empty.Bounds(), result)
	}

	// A non-zero-origin empty image must keep its bounds.
	shifted := image.NewRGBA(image.Rect(2, 3, 2, 6))
	result, err = SauvolaThreshold(shifted, 3, 0.2, 128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Bounds().Eq(shifted.Bounds()) {
		t.Fatalf("expected bounds %v, got %v", shifted.Bounds(), result.Bounds())
	}
}

func TestSauvolaThresholdDoesNotModifyInput(t *testing.T) {
	src := gradientLineArt(9, 5)
	before := make([]uint8, len(src.Pix))
	copy(before, src.Pix)

	if _, err := SauvolaThreshold(src, 3, 0.2, 128); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(before) != len(src.Pix) {
		t.Fatal("input pixel buffer length changed")
	}
	for i := range before {
		if before[i] != src.Pix[i] {
			t.Fatalf("input modified at byte %d: got %#x want %#x", i, src.Pix[i], before[i])
		}
	}
}

func TestSauvolaThresholdDeterministic(t *testing.T) {
	src := gradientLineArt(21, 11)
	first, err := SauvolaThreshold(src, 7, 0.2, 128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	second, err := SauvolaThreshold(src, 7, 0.2, 128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !util.GrayImageEqual(first, second) {
		t.Fatal("repeated runs produced different results")
	}
}

func BenchmarkSauvolaThreshold(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 500, 500))
	for y := 0; y < 500; y++ {
		for x := 0; x < 500; x++ {
			pos := src.PixOffset(x, y)
			v := uint8((x + y) % 256)
			src.Pix[pos+0], src.Pix[pos+1], src.Pix[pos+2] = v, v, v
			src.Pix[pos+3] = 0xFF
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SauvolaThreshold(src, DefaultSauvolaWindow, DefaultSauvolaK, DefaultSauvolaR); err != nil {
			b.Fatal(err)
		}
	}
}
