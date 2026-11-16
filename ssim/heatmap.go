package ssim

import "image"

// heatRamp is a single-hue sequential ramp (light to dark blue) used by
// ColorizeHeatmap. Light steps mean a near-perfect local score and dark steps
// mean damage, matching the grayscale heatmap's brightness convention.
var heatRamp = [][3]uint8{
	{0xCD, 0xE2, 0xFB},
	{0xB7, 0xD3, 0xF6},
	{0x9E, 0xC5, 0xF4},
	{0x86, 0xB6, 0xEF},
	{0x6D, 0xA7, 0xEC},
	{0x55, 0x98, 0xE7},
	{0x39, 0x87, 0xE5},
	{0x2A, 0x78, 0xD6},
	{0x25, 0x6A, 0xBF},
	{0x1C, 0x5C, 0xAB},
	{0x18, 0x4F, 0x95},
	{0x10, 0x42, 0x81},
	{0x0D, 0x36, 0x6B},
}

// ColorizeHeatmap returns a colorized copy of a quality heatmap using a
// single-hue sequential ramp. Light blue marks intact areas and dark blue
// marks local damage, so problem regions stand out at a glance without
// relying on hue alone.
//
// The returned image keeps the heatmap's bounds and is fully opaque. A nil
// heatmap yields an empty image. Pixels are mapped through a 13-step ramp and
// adjacent values are blended linearly, so the result stays smooth.
func ColorizeHeatmap(heatmap *image.Gray) *image.RGBA {
	if heatmap == nil {
		return &image.RGBA{}
	}

	bounds := heatmap.Bounds()
	dst := image.NewRGBA(bounds)
	if bounds.Empty() {
		return dst
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b := rampColor(heatmap.GrayAt(x, y).Y)
			pos := dst.PixOffset(x, y)
			dst.Pix[pos+0] = r
			dst.Pix[pos+1] = g
			dst.Pix[pos+2] = b
			dst.Pix[pos+3] = 0xFF
		}
	}

	return dst
}

// rampColor maps a quality value to the sequential ramp. A value of 255 is the
// lightest step and 0 the darkest; the ramp index is reversed accordingly.
func rampColor(quality uint8) (uint8, uint8, uint8) {
	damage := 255 - int(quality)
	steps := len(heatRamp) - 1

	// Position along the ramp in [0, steps], then blend the two neighbors.
	position := float64(damage) * float64(steps) / 255
	lower := int(position)
	if lower >= steps {
		c := heatRamp[steps]
		return c[0], c[1], c[2]
	}
	frac := position - float64(lower)
	a, b := heatRamp[lower], heatRamp[lower+1]

	blend := func(i int) uint8 {
		return uint8(float64(a[i]) + (float64(b[i])-float64(a[i]))*frac + 0.5)
	}
	return blend(0), blend(1), blend(2)
}
