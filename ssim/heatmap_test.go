package ssim

import (
	"image"
	"image/color"
	"testing"
)

func TestColorizeHeatmap(t *testing.T) {
	heat := image.NewGray(image.Rect(0, 0, 256, 1))
	for x := 0; x < 256; x++ {
		heat.Pix[x] = uint8(x)
	}

	result := ColorizeHeatmap(heat)
	if !result.Bounds().Eq(heat.Bounds()) {
		t.Fatalf("bounds changed: got %v, want %v", result.Bounds(), heat.Bounds())
	}

	for x := 0; x < 256; x++ {
		c := result.RGBAAt(x, 0)
		if c.A != 0xFF {
			t.Fatalf("output must be opaque at x=%d, got alpha %d", x, c.A)
		}
		if x == 0 {
			continue
		}
		prev := result.RGBAAt(x-1, 0)
		// Quality increases with x, so the ramp must never get darker.
		if luminance(c) < luminance(prev) {
			t.Fatalf("ramp is not monotonic at quality %d: %v is darker than %v", x, c, prev)
		}
	}

	// A perfect score is the lightest step, a zero score the darkest.
	if got := result.RGBAAt(255, 0); got != (color.RGBA{0xCD, 0xE2, 0xFB, 0xFF}) {
		t.Fatalf("unexpected color for a perfect score: %v", got)
	}
	if got := result.RGBAAt(0, 0); got != (color.RGBA{0x0D, 0x36, 0x6B, 0xFF}) {
		t.Fatalf("unexpected color for a zero score: %v", got)
	}

	empty := ColorizeHeatmap(nil)
	if !empty.Bounds().Empty() {
		t.Fatalf("nil heatmap must yield an empty image, got %v", empty.Bounds())
	}
}

func luminance(c color.RGBA) float64 {
	return 0.3*float64(c.R) + 0.6*float64(c.G) + 0.1*float64(c.B)
}

func TestColorizeHeatmapNonZeroOrigin(t *testing.T) {
	heat := image.NewGray(image.Rect(5, 7, 9, 10))
	heat.SetGray(5, 7, color.Gray{Y: 255})
	heat.SetGray(8, 9, color.Gray{Y: 0})

	result := ColorizeHeatmap(heat)
	if !result.Bounds().Eq(heat.Bounds()) {
		t.Fatalf("bounds changed: got %v, want %v", result.Bounds(), heat.Bounds())
	}
	if got := result.RGBAAt(5, 7); got != (color.RGBA{0xCD, 0xE2, 0xFB, 0xFF}) {
		t.Fatalf("unexpected color at the origin: %v", got)
	}
	if got := result.RGBAAt(8, 9); got != (color.RGBA{0x0D, 0x36, 0x6B, 0xFF}) {
		t.Fatalf("unexpected color at the far corner: %v", got)
	}
}
