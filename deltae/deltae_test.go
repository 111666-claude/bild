package deltae

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// Reference color pairs from "The CIEDE2000 Color-Difference Formula:
// Implementation Notes, Supplementary Test Data, and Mathematical
// Observations" (Sharma, Wu and Dalal, 2005), Table 1.
var sharmaTestPairs = []struct {
	l1, a1, b1 float64
	l2, a2, b2 float64
	expected   float64
}{
	{50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425},
	{50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615},
	{50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412},
	{50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000},
	{50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000},
	{50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000},
	{50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669},
	{50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669},
	{50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792},
	{50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792},
	{50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195},
	{50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195},
	{50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045},
	{50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045},
	{50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461},
	{50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065},
	{50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492},
	{50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977},
	{50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030},
	{50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535},
	{50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000},
	{50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000},
	{50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000},
	{50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000},
	{60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644},
	{63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630},
	{61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731},
	{35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645},
	{22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373},
	{36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146},
	{90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441},
	{90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381},
	{6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377},
	{2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082},
}

func TestDeltaE2000ReferencePairs(t *testing.T) {
	for i, c := range sharmaTestPairs {
		got := deltaE2000(
			labColor{l: c.l1, a: c.a1, b: c.b1},
			labColor{l: c.l2, a: c.a2, b: c.b2},
		)
		if math.Abs(got-c.expected) > 0.0001 {
			t.Errorf("pair %d: expected %.4f, got %.4f", i, c.expected, got)
		}
	}
}

func TestDeltaEIdenticalColors(t *testing.T) {
	colors := []color.NRGBA{
		{0, 0, 0, 0xFF},
		{0xFF, 0xFF, 0xFF, 0xFF},
		{0x80, 0x40, 0xC0, 0xFF},
		{0x12, 0x34, 0x56, 0x80},
		{0xFF, 0x00, 0x00, 0xFF},
		{0x00, 0xFF, 0xFF, 0xFF},
	}
	for _, c := range colors {
		if d := DeltaE(c, c); d != 0 {
			t.Errorf("expected zero difference for identical color %#v, got %v", c, d)
		}
	}
}

func TestDeltaEAchromaticRange(t *testing.T) {
	// Black to white is a pure lightness difference of 100.
	d := DeltaE(color.NRGBA{0, 0, 0, 0xFF}, color.NRGBA{0xFF, 0xFF, 0xFF, 0xFF})
	if math.Abs(d-100) > 0.01 {
		t.Errorf("expected black-white difference of 100, got %v", d)
	}
}

func TestDeltaENeverNegativeOrNaN(t *testing.T) {
	// Exercise extreme and near-identical colors, including cases where the
	// CIEDE2000 rotation term can push the squared sum below zero.
	steps := []uint8{0, 1, 2, 64, 128, 192, 254, 255}
	for _, r1 := range steps {
		for _, g1 := range steps {
			for _, b1 := range steps {
				c1 := color.NRGBA{r1, g1, b1, 0xFF}
				for _, r2 := range []uint8{0, 128, 255} {
					c2 := color.NRGBA{r2, g1, b1, 0xFF}
					d := DeltaE(c1, c2)
					if math.IsNaN(d) || d < 0 {
						t.Fatalf("invalid DeltaE %v for %#v vs %#v", d, c1, c2)
					}
				}
			}
		}
	}
}

func TestCompareIdenticalImages(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for i := range img.Pix {
		img.Pix[i] = uint8((i * 37) % 256)
	}

	result, err := Compare(img, img)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Mean != 0 || result.Max != 0 {
		t.Errorf("expected zero mean and max, got mean=%v max=%v", result.Mean, result.Max)
	}
	for i, v := range result.Values {
		if v != 0 {
			t.Errorf("expected zero value at index %d, got %v", i, v)
		}
	}
	for i := 0; i < len(result.Heatmap.Pix); i += 4 {
		if result.Heatmap.Pix[i] != 0 || result.Heatmap.Pix[i+1] != 0 || result.Heatmap.Pix[i+2] != 0 {
			t.Errorf("expected black heatmap pixel at %d", i)
		}
	}
}

func TestComparePreservesBoundsAndAlpha(t *testing.T) {
	bounds := image.Rect(3, 7, 8, 11)
	a := image.NewNRGBA(bounds)
	b := image.NewNRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			a.SetNRGBA(x, y, color.NRGBA{0x10, 0x20, 0x30, uint8(50 + x*20 + y)})
			b.SetNRGBA(x, y, color.NRGBA{0x40, 0x50, 0x60, 0xFF})
		}
	}

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Heatmap.Bounds().Eq(bounds) {
		t.Errorf("expected heatmap bounds %v, got %v", bounds, result.Heatmap.Bounds())
	}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			expected := a.NRGBAAt(x, y).A
			if got := result.Heatmap.RGBAAt(x, y).A; got != expected {
				t.Errorf("alpha at (%d,%d): expected %d, got %d", x, y, expected, got)
			}
		}
	}
}

func TestCompareDoesNotMutateInputs(t *testing.T) {
	a := image.NewRGBA(image.Rect(0, 0, 2, 2))
	b := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := range a.Pix {
		a.Pix[i] = uint8(i * 3)
		b.Pix[i] = uint8(i * 5)
	}
	aCopy := append([]uint8(nil), a.Pix...)
	bCopy := append([]uint8(nil), b.Pix...)

	if _, err := Compare(a, b); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i := range a.Pix {
		if a.Pix[i] != aCopy[i] || b.Pix[i] != bCopy[i] {
			t.Fatal("input image was modified")
		}
	}
}

func TestCompareTransparentPixelsExcludedFromSummary(t *testing.T) {
	a := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	b := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	// One opaque pixel, identical in both images.
	a.SetNRGBA(0, 0, color.NRGBA{0x80, 0x80, 0x80, 0xFF})
	b.SetNRGBA(0, 0, color.NRGBA{0x80, 0x80, 0x80, 0xFF})
	// One fully transparent pixel with wildly different stored colors.
	a.SetNRGBA(1, 0, color.NRGBA{0xFF, 0x00, 0x00, 0x00})
	b.SetNRGBA(1, 0, color.NRGBA{0x00, 0x00, 0xFF, 0x00})

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Mean != 0 || result.Max != 0 {
		t.Errorf("transparent pixel leaked into summary: mean=%v max=%v", result.Mean, result.Max)
	}
	// The per-pixel value is still computed, but the heatmap alpha is preserved.
	if result.At(1, 0) <= 0 {
		t.Errorf("expected non-zero per-pixel value for differing transparent pixel")
	}
	if got := result.Heatmap.RGBAAt(1, 0).A; got != 0 {
		t.Errorf("expected heatmap alpha 0, got %d", got)
	}
}

func TestCompareFullyTransparentImages(t *testing.T) {
	a := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	b := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			a.SetNRGBA(x, y, color.NRGBA{0xFF, 0x00, 0x00, 0x00})
			b.SetNRGBA(x, y, color.NRGBA{0x00, 0xFF, 0x00, 0x00})
		}
	}

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if math.IsNaN(result.Mean) || result.Mean != 0 {
		t.Errorf("expected zero mean for fully transparent images, got %v", result.Mean)
	}
	if math.IsNaN(result.Max) || result.Max != 0 {
		t.Errorf("expected zero max for fully transparent images, got %v", result.Max)
	}
}

func TestCompareConsistentAcrossImageTypes(t *testing.T) {
	bounds := image.Rect(0, 0, 4, 4)
	nrgba := image.NewNRGBA(bounds)
	rgba := image.NewRGBA(bounds)
	gray := image.NewGray(bounds)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			c := color.NRGBA{uint8(x * 60), uint8(y * 60), 0x80, 0xFF}
			nrgba.SetNRGBA(x, y, c)
			rgba.Set(x, y, c)
			gray.SetGray(x, y, color.Gray{uint8(x*20 + y*20)})
		}
	}

	ref, err := Compare(nrgba, gray)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other, err := Compare(rgba, gray)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ref.Mean != other.Mean || ref.Max != other.Max {
		t.Errorf("inconsistent results across image types: %+v vs %+v", ref, other)
	}
	for i := range ref.Values {
		if ref.Values[i] != other.Values[i] {
			t.Fatalf("inconsistent per-pixel value at %d: %v vs %v", i, ref.Values[i], other.Values[i])
		}
	}
}

func TestCompareAlphaWeightedMean(t *testing.T) {
	a := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	b := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	// Pixel 0: opaque, colors differ. Pixel 1: half alpha, same difference.
	a.SetNRGBA(0, 0, color.NRGBA{0x00, 0x00, 0x00, 0xFF})
	b.SetNRGBA(0, 0, color.NRGBA{0xFF, 0xFF, 0xFF, 0xFF})
	a.SetNRGBA(1, 0, color.NRGBA{0x00, 0x00, 0x00, 0x80})
	b.SetNRGBA(1, 0, color.NRGBA{0xFF, 0xFF, 0xFF, 0x80})

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	d := DeltaE(color.NRGBA{0, 0, 0, 0xFF}, color.NRGBA{0xFF, 0xFF, 0xFF, 0xFF})
	w0 := 1.0
	w1 := (float64(0x80) + float64(0x80)) / (2 * 255)
	expected := (w0*d + w1*d) / (w0 + w1)
	if math.Abs(result.Mean-expected) > 1e-9 {
		t.Errorf("expected mean %v, got %v", expected, result.Mean)
	}
	if math.Abs(result.Mean-d) > 1e-9 {
		t.Errorf("expected weighted mean to equal the shared difference %v, got %v", d, result.Mean)
	}
}

func TestCompareSyntheticLocalChange(t *testing.T) {
	const size = 8
	a := image.NewNRGBA(image.Rect(0, 0, size, size))
	b := image.NewNRGBA(image.Rect(0, 0, size, size))
	base := color.NRGBA{0x60, 0x90, 0xC0, 0xFF}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			a.SetNRGBA(x, y, base)
			b.SetNRGBA(x, y, base)
		}
	}
	// Locally recolor a 2x2 hot region in b.
	hot := color.NRGBA{0xFF, 0x40, 0x20, 0xFF}
	for y := 3; y < 5; y++ {
		for x := 3; x < 5; x++ {
			b.SetNRGBA(x, y, hot)
		}
	}

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedDelta := DeltaE(base, hot)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			inHotRegion := x >= 3 && x < 5 && y >= 3 && y < 5
			got := result.At(x, y)
			if inHotRegion {
				if math.Abs(got-expectedDelta) > 1e-9 {
					t.Errorf("hot region (%d,%d): expected %v, got %v", x, y, expectedDelta, got)
				}
				intensity := result.Heatmap.RGBAAt(x, y).R
				if intensity == 0 {
					t.Errorf("hot region (%d,%d): expected non-zero heatmap intensity", x, y)
				}
			} else {
				if got != 0 {
					t.Errorf("unchanged pixel (%d,%d): expected zero delta, got %v", x, y, got)
				}
				if result.Heatmap.RGBAAt(x, y).R != 0 {
					t.Errorf("unchanged pixel (%d,%d): expected zero heatmap intensity", x, y)
				}
			}
		}
	}

	expectedMean := expectedDelta * 4 / (size * size)
	if math.Abs(result.Mean-expectedMean) > 1e-9 {
		t.Errorf("expected mean %v, got %v", expectedMean, result.Mean)
	}
	if math.Abs(result.Max-expectedDelta) > 1e-9 {
		t.Errorf("expected max %v, got %v", expectedDelta, result.Max)
	}
}

func TestCompareDifferentOrigins(t *testing.T) {
	a := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	b := image.NewNRGBA(image.Rect(10, 20, 12, 22))
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			a.SetNRGBA(x, y, color.NRGBA{0x11, 0x22, 0x33, 0xFF})
			b.SetNRGBA(10+x, 20+y, color.NRGBA{0x11, 0x22, 0x33, 0xFF})
		}
	}

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Mean != 0 || result.Max != 0 {
		t.Errorf("expected zero difference for same content at different origins, got mean=%v max=%v", result.Mean, result.Max)
	}
	if !result.Heatmap.Bounds().Eq(image.Rect(0, 0, 2, 2)) {
		t.Errorf("expected heatmap bounds to follow first input, got %v", result.Heatmap.Bounds())
	}
}

func TestCompareInvalidInputs(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))

	if _, err := Compare(nil, img); err != ErrNilImage {
		t.Errorf("expected ErrNilImage, got %v", err)
	}
	if _, err := Compare(img, nil); err != ErrNilImage {
		t.Errorf("expected ErrNilImage, got %v", err)
	}

	other := image.NewRGBA(image.Rect(0, 0, 3, 2))
	if _, err := Compare(img, other); err != ErrSizeMismatch {
		t.Errorf("expected ErrSizeMismatch, got %v", err)
	}
}

func TestCompareEmptyImages(t *testing.T) {
	a := image.NewRGBA(image.Rectangle{})
	b := image.NewRGBA(image.Rectangle{})

	result, err := Compare(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Mean != 0 || result.Max != 0 || len(result.Values) != 0 {
		t.Errorf("expected zero result for empty images, got %+v", result)
	}
	if math.IsNaN(result.Mean) || math.IsNaN(result.Max) {
		t.Error("unexpected NaN in empty image result")
	}
}

func TestCompareInvalidScale(t *testing.T) {
	a := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	b := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	a.SetNRGBA(0, 0, color.NRGBA{0x00, 0x00, 0x00, 0xFF})
	b.SetNRGBA(0, 0, color.NRGBA{0xFF, 0xFF, 0xFF, 0xFF})

	for _, scale := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		result, err := CompareWithScale(a, b, scale)
		if err != nil {
			t.Fatalf("scale %v: unexpected error: %v", scale, err)
		}
		if math.IsNaN(result.Mean) || result.Mean < 0 {
			t.Errorf("scale %v: invalid mean %v", scale, result.Mean)
		}
		// Black vs white is a difference of 100, which is exactly full
		// intensity at the default scale.
		if got := result.Heatmap.RGBAAt(0, 0).R; got != 0xFF {
			t.Errorf("scale %v: expected full heatmap intensity, got %d", scale, got)
		}
	}

	// A valid custom scale changes the normalization.
	result, err := CompareWithScale(a, b, 200)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := result.Heatmap.RGBAAt(0, 0).R; got != 0x80 {
		t.Errorf("expected half intensity at scale 200, got %d", got)
	}
}

func BenchmarkCompare(b *testing.B) {
	imgA := image.NewRGBA(image.Rect(0, 0, 512, 512))
	imgB := image.NewRGBA(image.Rect(0, 0, 512, 512))
	for i := range imgA.Pix {
		imgA.Pix[i] = uint8(i * 3)
		imgB.Pix[i] = uint8(i * 7)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult, _ = Compare(imgA, imgB)
	}
}

// benchResult is used to avoid having the compiler optimize the benchmark code calls
var benchResult interface{}
