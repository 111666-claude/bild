package halftone

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/anthonynsimon/bild/transform"
	"github.com/anthonynsimon/bild/util"
)

func nan() float64 { return math.NaN() }

func inf() float64 { return math.Inf(1) }

// uniformRGBA builds a width x height image filled with the given color.
func uniformRGBA(width, height int, c color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pos := y*img.Stride + x*4
			img.Pix[pos+0] = c.R
			img.Pix[pos+1] = c.G
			img.Pix[pos+2] = c.B
			img.Pix[pos+3] = c.A
		}
	}
	return img
}

func TestBayerMatrix(t *testing.T) {
	cases := []struct {
		size     int
		expected []float64
	}{
		{
			size: 1,
			// The single threshold level sits at mid-gray.
			expected: []float64{128},
		},
		{
			size: 2,
			// Classic 2x2 Bayer matrix centered within the (0, 255) range.
			expected: []float64{
				32, 160,
				224, 96,
			},
		},
		{
			size: 4,
			// Canonical 4x4 Bayer matrix (rank 0..15), centered and scaled to
			// the (0, 255) range with a 16-cell spacing of 16.
			expected: []float64{
				8, 40, 136, 168,
				56, 24, 184, 152,
				200, 232, 72, 104,
				248, 216, 120, 88,
			},
		},
	}

	for _, c := range cases {
		actual, err := BayerMatrix(c.size)
		if err != nil {
			t.Fatalf("BayerMatrix(%d): unexpected error %v", c.size, err)
		}
		if len(actual) != len(c.expected) {
			t.Fatalf("BayerMatrix(%d): expected %d values, got %d", c.size, len(c.expected), len(actual))
		}
		for i := range actual {
			if actual[i] != c.expected[i] {
				t.Errorf("BayerMatrix(%d) at %d: expected %v, got %v", c.size, i, c.expected, actual)
				break
			}
		}
	}
}

func TestBayerMatrixInvalid(t *testing.T) {
	for _, size := range []int{0, -1, -8, MaxMatrixSize + 1, 1000} {
		if _, err := BayerMatrix(size); err != ErrInvalidMatrixSize {
			t.Errorf("BayerMatrix(%d): expected ErrInvalidMatrixSize, got %v", size, err)
		}
	}
}

func TestBayerMatrixDistinctLevels(t *testing.T) {
	// Every supported size must produce size*size distinct threshold levels.
	for size := 1; size <= 17; size++ {
		matrix, err := BayerMatrix(size)
		if err != nil {
			t.Fatalf("BayerMatrix(%d): unexpected error %v", size, err)
		}
		seen := make(map[float64]bool, len(matrix))
		for _, v := range matrix {
			if seen[v] {
				t.Fatalf("BayerMatrix(%d): duplicated threshold level %v", size, v)
			}
			seen[v] = true
			if v <= 0 || v >= 256 {
				t.Fatalf("BayerMatrix(%d): threshold %v out of the (0, 256) range", size, v)
			}
		}
	}
}

func TestOrderedFixedGrayscale(t *testing.T) {
	// A uniform mid-gray (128) image dithered with a 2x2 matrix. Lum is
	// exactly 128, so a pixel turns white only where its centered threshold
	// is above 127.5 (the two highest matrix ranks).
	img := uniformRGBA(4, 4, color.RGBA{128, 128, 128, 255})

	actual, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("Ordered: unexpected error %v", err)
	}

	expected := image.NewRGBA(image.Rect(0, 0, 4, 4))
	// Canonical 2x2 Bayer tile (W = white, B = black) for a lum of exactly
	// 128: white appears at the two highest matrix ranks (threshold > 127.5):
	//	B W
	//	W B
	pattern := [2][2]bool{{false, true}, {true, false}}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			pos := y*expected.Stride + x*4
			if pattern[y%2][x%2] {
				expected.Pix[pos+0], expected.Pix[pos+1], expected.Pix[pos+2] = 255, 255, 255
			}
			expected.Pix[pos+3] = 255
		}
	}

	if !util.RGBAImageEqual(actual, expected) {
		t.Errorf("Ordered mid-gray dither mismatch.\nexpected:%v\nactual:%v",
			util.RGBAToString(expected), util.RGBAToString(actual))
	}
}

func TestOrderedSolidBlackAndWhite(t *testing.T) {
	// Extreme tones must stay solid regardless of the matrix thresholds.
	black := uniformRGBA(3, 3, color.RGBA{0, 0, 0, 255})
	white := uniformRGBA(3, 3, color.RGBA{255, 255, 255, 255})

	outBlack, err := Ordered(black, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("Ordered black: unexpected error %v", err)
	}
	outWhite, err := Ordered(white, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("Ordered white: unexpected error %v", err)
	}

	for y := 0; y < 3; y++ {
		for x := 0; x < 4*3; x += 4 {
			pos := y*outBlack.Stride + x
			if outBlack.Pix[pos] != 0 || outBlack.Pix[pos+1] != 0 || outBlack.Pix[pos+2] != 0 || outBlack.Pix[pos+3] != 255 {
				t.Fatalf("black input produced non-black pixel at (%d,%d): %v", x/4, y, outBlack.Pix[pos:pos+4])
			}
			pos = y*outWhite.Stride + x
			if outWhite.Pix[pos] != 255 || outWhite.Pix[pos+1] != 255 || outWhite.Pix[pos+2] != 255 || outWhite.Pix[pos+3] != 255 {
				t.Fatalf("white input produced non-white pixel at (%d,%d): %v", x/4, y, outWhite.Pix[pos:pos+4])
			}
		}
	}
}

func TestOrderedTransparentCheckerboard(t *testing.T) {
	// Checkerboard of fully transparent and opaque gray pixels. Transparent
	// cells must remain fully transparent; opaque cells are dithered normally.
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			pos := y*img.Stride + x*4
			if (x+y)%2 == 0 {
				// Fully transparent pixel: RGB channels deliberately non-zero
				// to prove only alpha drives the transparency rule.
				img.Pix[pos+0], img.Pix[pos+1], img.Pix[pos+2], img.Pix[pos+3] = 200, 100, 50, 0
			} else {
				img.Pix[pos+0], img.Pix[pos+1], img.Pix[pos+2], img.Pix[pos+3] = 128, 128, 128, 255
			}
		}
	}

	actual, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("Ordered: unexpected error %v", err)
	}

	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			pos := y*actual.Stride + x*4
			got := actual.Pix[pos : pos+4]
			if (x+y)%2 == 0 {
				if got[0] != 0 || got[1] != 0 || got[2] != 0 || got[3] != 0 {
					t.Errorf("transparent pixel at (%d,%d) not preserved: %v", x, y, got)
				}
			} else if got[3] != 255 {
				t.Errorf("opaque pixel at (%d,%d) lost opacity: %v", x, y, got)
			}
		}
	}
}

func TestOrderedCropConsistency(t *testing.T) {
	// The matrix is indexed by absolute coordinates, so processing a cropped
	// block must equal the corresponding region of the full-image result.
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			pos := y*img.Stride + x*4
			v := uint8((x*17 + y*31) % 256)
			img.Pix[pos+0], img.Pix[pos+1], img.Pix[pos+2], img.Pix[pos+3] = v, v, v, 255
		}
	}

	opts := &Options{MatrixSize: 3, Strength: 0.75}
	full, err := Ordered(img, opts)
	if err != nil {
		t.Fatalf("Ordered full: unexpected error %v", err)
	}

	cropRect := image.Rect(2, 1, 6, 5)
	cropped := transform.Crop(img, cropRect)
	// The crop keeps the original absolute coordinate origin.
	if cropped.Bounds().Min != cropRect.Min {
		t.Fatalf("crop bounds min expected %v, got %v", cropRect.Min, cropped.Bounds().Min)
	}

	part, err := Ordered(cropped, opts)
	if err != nil {
		t.Fatalf("Ordered crop: unexpected error %v", err)
	}

	if part.Bounds() != cropRect {
		t.Fatalf("part bounds expected %v, got %v", cropRect, part.Bounds())
	}

	// Compare through At so the check is independent of how each image's
	// Pix buffer is based relative to its bounds.
	for y := cropRect.Min.Y; y < cropRect.Max.Y; y++ {
		for x := cropRect.Min.X; x < cropRect.Max.X; x++ {
			if full.At(x, y) != part.At(x, y) {
				t.Fatalf("crop mismatch at (%d,%d): full=%v part=%v",
					x, y, full.At(x, y), part.At(x, y))
			}
		}
	}
}

func TestOrderedNonZeroBounds(t *testing.T) {
	// A hand-built image whose origin is not (0,0) must be processed without
	// out-of-range indexing and keep the same bounds.
	rect := image.Rect(5, 7, 8, 10)
	src := image.NewRGBA(rect)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			// SetRGBA addresses pixels by logical coordinates, so this stays
			// correct even though the origin is not (0,0).
			src.SetRGBA(x, y, color.RGBA{128, 128, 128, 255})
		}
	}

	actual, err := Ordered(src, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("Ordered: unexpected error %v", err)
	}
	if actual.Bounds() != rect {
		t.Fatalf("bounds expected %v, got %v", rect, actual.Bounds())
	}
	// The Pix buffer covers exactly the 3x3 region regardless of its origin.
	if actual.Rect.Dx() != 3 || actual.Rect.Dy() != 3 {
		t.Fatalf("unexpected dimensions: %v", actual.Rect)
	}

	// Reference: same logical pixels processed at origin (0,0) with an Origin
	// shift of (5,7) must match, proving absolute-coordinate indexing.
	shifted := uniformRGBA(3, 3, color.RGBA{128, 128, 128, 255})
	ref, err := Ordered(shifted, &Options{MatrixSize: 2, Strength: 1, Origin: image.Pt(5, 7)})
	if err != nil {
		t.Fatalf("Ordered reference: unexpected error %v", err)
	}
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			if actual.At(5+x, 7+y) != ref.At(x, y) {
				t.Fatalf("non-zero bounds mismatch at (%d,%d): %v vs %v",
					x, y, actual.At(5+x, 7+y), ref.At(x, y))
			}
		}
	}
}

func TestOrderedOriginShift(t *testing.T) {
	img := uniformRGBA(4, 4, color.RGBA{128, 128, 128, 255})

	base, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("Ordered base: unexpected error %v", err)
	}
	shifted, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1, Origin: image.Pt(1, 0)})
	if err != nil {
		t.Fatalf("Ordered shifted: unexpected error %v", err)
	}

	// A one-cell horizontal shift must change the pattern (and not panic on
	// negative modulo math).
	different := false
	for i := range base.Pix {
		if base.Pix[i] != shifted.Pix[i] {
			different = true
			break
		}
	}
	if !different {
		t.Error("Origin shift had no effect on the dither pattern")
	}

	// Shifting by a full matrix period must reproduce the original pattern.
	period, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1, Origin: image.Pt(2, 2)})
	if err != nil {
		t.Fatalf("Ordered period: unexpected error %v", err)
	}
	if !util.RGBAImageEqual(base, period) {
		t.Error("a full-period Origin shift should reproduce the base pattern")
	}

	// Negative origins must be handled via Euclidean modulo.
	neg, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1, Origin: image.Pt(-3, -5)})
	if err != nil {
		t.Fatalf("Ordered negative origin: unexpected error %v", err)
	}
	if neg.Bounds() != img.Bounds() {
		t.Fatalf("negative origin changed bounds: %v", neg.Bounds())
	}
}

func TestOrderedCustomPalette(t *testing.T) {
	img := uniformRGBA(2, 2, color.RGBA{128, 128, 128, 255})
	palette := color.Palette{
		color.RGBA{0, 0, 0, 255},
		color.RGBA{255, 0, 0, 255},
		color.RGBA{255, 255, 255, 255},
	}

	actual, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1, Palette: palette})
	if err != nil {
		t.Fatalf("Ordered: unexpected error %v", err)
	}

	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			pos := y*actual.Stride + x*4
			got := color.RGBA{actual.Pix[pos], actual.Pix[pos+1], actual.Pix[pos+2], actual.Pix[pos+3]}
			if !paletteContains(palette, got) {
				t.Errorf("pixel (%d,%d) = %v is not part of the palette", x, y, got)
			}
		}
	}
}

func paletteContains(palette color.Palette, c color.RGBA) bool {
	for _, p := range palette {
		r, g, b, a := p.RGBA()
		if uint8(r>>8) == c.R && uint8(g>>8) == c.G && uint8(b>>8) == c.B && uint8(a>>8) == c.A {
			return true
		}
	}
	return false
}

func TestOrderedStrengthZero(t *testing.T) {
	// A vanishingly small strength makes the matrix effect negligible: every
	// pixel is quantized to the nearest palette entry, so a uniform image
	// stays uniform.
	img := uniformRGBA(4, 4, color.RGBA{128, 128, 128, 255})
	actual, err := Ordered(img, &Options{MatrixSize: 4, Strength: 1e-9})
	if err != nil {
		t.Fatalf("Ordered: unexpected error %v", err)
	}

	first := actual.Pix[0:4]
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			pos := y*actual.Stride + x*4
			for c := 0; c < 4; c++ {
				if actual.Pix[pos+c] != first[c] {
					t.Fatalf("near-zero strength produced non-uniform output at (%d,%d)", x, y)
				}
			}
		}
	}
	// Lum 128 is closer to white (distance 127) than black (distance 128),
	// so white wins.
	if first[0] != 255 {
		t.Errorf("near-zero strength mid-gray expected white, got %v", first)
	}
}

func TestOrderedUnsetStrengthDefaultsToFull(t *testing.T) {
	// The zero value of Strength means "unset" and must select full strength,
	// identical to an explicit 1.0.
	img := uniformRGBA(4, 4, color.RGBA{128, 128, 128, 255})

	unset, err := Ordered(img, &Options{MatrixSize: 2})
	if err != nil {
		t.Fatalf("unset strength: unexpected error %v", err)
	}
	full, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("full strength: unexpected error %v", err)
	}
	if !util.RGBAImageEqual(unset, full) {
		t.Error("unset Strength should default to full strength")
	}
}

func TestOrderedStrengthClamping(t *testing.T) {
	img := uniformRGBA(4, 4, color.RGBA{128, 128, 128, 255})

	negative, err := Ordered(img, &Options{MatrixSize: 2, Strength: -5})
	if err != nil {
		t.Fatalf("negative strength: unexpected error %v", err)
	}
	tiny, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1e-9})
	if err != nil {
		t.Fatalf("tiny strength: unexpected error %v", err)
	}
	if !util.RGBAImageEqual(negative, tiny) {
		t.Error("negative strength should be clamped to zero")
	}

	huge, err := Ordered(img, &Options{MatrixSize: 2, Strength: 100})
	if err != nil {
		t.Fatalf("huge strength: unexpected error %v", err)
	}
	one, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1})
	if err != nil {
		t.Fatalf("unit strength: unexpected error %v", err)
	}
	if !util.RGBAImageEqual(huge, one) {
		t.Error("strength above 1 should be clamped to 1")
	}
}

func TestOrderedEmptyImage(t *testing.T) {
	actual, err := Ordered(&image.RGBA{}, nil)
	if err != nil {
		t.Fatalf("empty image: unexpected error %v", err)
	}
	if !actual.Bounds().Empty() {
		t.Errorf("expected empty bounds, got %v", actual.Bounds())
	}

	// Empty bounds with a non-zero origin are preserved.
	rect := image.Rect(3, 4, 3, 9)
	actual, err = Ordered(image.NewRGBA(rect), nil)
	if err != nil {
		t.Fatalf("empty non-zero bounds: unexpected error %v", err)
	}
	if actual.Bounds() != rect {
		t.Errorf("expected bounds %v, got %v", rect, actual.Bounds())
	}
}

func TestOrderedTinyImage(t *testing.T) {
	// A 1x1 image, including a matrix larger than the image itself.
	img := uniformRGBA(1, 1, color.RGBA{128, 128, 128, 255})
	actual, err := Ordered(img, &Options{MatrixSize: 8, Strength: 1})
	if err != nil {
		t.Fatalf("tiny image: unexpected error %v", err)
	}
	if actual.Bounds() != image.Rect(0, 0, 1, 1) {
		t.Fatalf("unexpected bounds %v", actual.Bounds())
	}
	got := actual.Pix[0]
	if got != 0 && got != 255 {
		t.Errorf("1x1 output expected black or white, got %d", got)
	}
}

func TestOrderedMaxMatrixSize(t *testing.T) {
	// The largest supported matrix on a small image must work, and one past
	// the limit must be rejected.
	img := uniformRGBA(2, 2, color.RGBA{64, 64, 64, 255})
	actual, err := Ordered(img, &Options{MatrixSize: MaxMatrixSize, Strength: 1})
	if err != nil {
		t.Fatalf("max matrix size: unexpected error %v", err)
	}
	if actual.Bounds() != img.Bounds() {
		t.Fatalf("unexpected bounds %v", actual.Bounds())
	}

	if _, err := Ordered(img, &Options{MatrixSize: MaxMatrixSize + 1, Strength: 1}); err != ErrInvalidMatrixSize {
		t.Errorf("oversized matrix: expected ErrInvalidMatrixSize, got %v", err)
	}
}

func TestOrderedDefaultOptions(t *testing.T) {
	img := uniformRGBA(4, 4, color.RGBA{128, 128, 128, 255})

	// nil options must work and default to a black/white palette, the 8x8
	// matrix and full strength.
	actual, err := Ordered(img, nil)
	if err != nil {
		t.Fatalf("nil options: unexpected error %v", err)
	}
	for i := 0; i < len(actual.Pix); i += 4 {
		r, g, b, a := actual.Pix[i], actual.Pix[i+1], actual.Pix[i+2], actual.Pix[i+3]
		if a != 255 || !(r == 0 && g == 0 && b == 0) && !(r == 255 && g == 255 && b == 255) {
			t.Fatalf("default output contains non black/white pixel: %v", actual.Pix[i:i+4])
		}
	}

	// Explicit zero MatrixSize must default to 8 as well.
	zeroSize, err := Ordered(img, &Options{MatrixSize: 0, Strength: 1})
	if err != nil {
		t.Fatalf("zero matrix size: unexpected error %v", err)
	}
	if !util.RGBAImageEqual(actual, zeroSize) {
		t.Error("zero MatrixSize should default to the 8x8 matrix")
	}
}

func TestOrderedErrors(t *testing.T) {
	img := uniformRGBA(2, 2, color.RGBA{128, 128, 128, 255})

	if _, err := Ordered(nil, nil); err != ErrNilImage {
		t.Errorf("nil image: expected ErrNilImage, got %v", err)
	}
	if _, err := Ordered(img, &Options{MatrixSize: -4, Strength: 1}); err != ErrInvalidMatrixSize {
		t.Errorf("negative matrix: expected ErrInvalidMatrixSize, got %v", err)
	}
	if _, err := Ordered(img, &Options{MatrixSize: 2, Strength: nan()}); err != ErrInvalidStrength {
		t.Errorf("NaN strength: expected ErrInvalidStrength, got %v", err)
	}
	if _, err := Ordered(img, &Options{MatrixSize: 2, Strength: inf()}); err != ErrInvalidStrength {
		t.Errorf("Inf strength: expected ErrInvalidStrength, got %v", err)
	}
	if _, err := Ordered(img, &Options{MatrixSize: 2, Strength: 1, Palette: color.Palette{}}); err != ErrInvalidPalette {
		t.Errorf("empty palette: expected ErrInvalidPalette, got %v", err)
	}
}
