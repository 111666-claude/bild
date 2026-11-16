/*
Package deltae provides CIEDE2000 color difference analysis between images.

Colors are converted from sRGB to CIE L*a*b* using the D65 illuminant, and the
perceptual difference is computed with the CIEDE2000 formula as defined in
"The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
Test Data, and Mathematical Observations" by Sharma, Wu and Dalal (2005).
*/
package deltae

import (
	"errors"
	"image"
	"image/color"
	"math"

	"github.com/anthonynsimon/bild/parallel"
)

// DefaultScale is the default DeltaE value that maps to full intensity in the
// heatmap returned by Compare. CIEDE2000 differences in common image content
// rarely exceed this value; larger differences are clamped to full intensity.
const DefaultScale = 100.0

// ErrSizeMismatch is returned when the two provided images do not have the
// same width and height.
var ErrSizeMismatch = errors.New("deltae: image sizes do not match")

// ErrNilImage is returned when a nil image is provided.
var ErrNilImage = errors.New("deltae: nil image")

// Result holds the outcome of a CIEDE2000 comparison between two images.
type Result struct {
	// Heatmap is a grayscale visualization of the per-pixel DeltaE values.
	// It has the same Bounds as the first input image. A pixel's intensity
	// is the pixel's DeltaE normalized by the comparison scale, and its
	// alpha channel is copied from the first input image.
	Heatmap *image.RGBA
	// Mean is the alpha-weighted average DeltaE over all pixels. Pixels
	// whose combined alpha weight is zero (fully transparent in both
	// inputs) do not contribute. It is zero when no pixel contributes.
	Mean float64
	// Max is the largest per-pixel DeltaE among pixels with a non-zero
	// alpha weight. It is zero when no pixel contributes.
	Max float64
	// Values holds the raw per-pixel DeltaE values in row-major order,
	// with Width*Height entries following the Heatmap bounds.
	Values []float64
	// Width and Height are the dimensions of Values and Heatmap.
	Width  int
	Height int
}

// At returns the raw per-pixel DeltaE value at the given position, expressed
// in Heatmap bounds coordinates. It returns 0 for out-of-bounds positions.
func (r *Result) At(x, y int) float64 {
	if r.Heatmap == nil {
		return 0
	}
	b := r.Heatmap.Bounds()
	ix, iy := x-b.Min.X, y-b.Min.Y
	if ix < 0 || iy < 0 || ix >= r.Width || iy >= r.Height {
		return 0
	}
	return r.Values[iy*r.Width+ix]
}

// Compare computes the per-pixel CIEDE2000 difference between images a and b
// using the DefaultScale for the heatmap normalization. See CompareWithScale.
func Compare(a, b image.Image) (*Result, error) {
	return CompareWithScale(a, b, DefaultScale)
}

// CompareWithScale computes the per-pixel CIEDE2000 difference between images
// a and b, which must have the same width and height (their origins may
// differ; pixels are matched by offset). The scale parameter is the DeltaE
// value that maps to full heatmap intensity; values <= 0, NaN or Inf fall
// back to DefaultScale.
//
// Colors are read as non-premultiplied sRGB, so different image.Image
// implementations of the same visual content produce identical results.
// Neither input is modified. The alpha weight of a pixel is the average of
// both images' alpha channels; fully transparent pixels are excluded from
// the Mean and Max summaries.
func CompareWithScale(a, b image.Image, scale float64) (*Result, error) {
	if a == nil || b == nil {
		return nil, ErrNilImage
	}

	bounds := a.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if b.Bounds().Dx() != w || b.Bounds().Dy() != h {
		return nil, ErrSizeMismatch
	}

	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		scale = DefaultScale
	}

	result := &Result{
		Width:  w,
		Height: h,
	}
	if w == 0 || h == 0 {
		result.Heatmap = image.NewRGBA(bounds)
		return result, nil
	}

	heatmap := image.NewRGBA(bounds)
	values := make([]float64, w*h)
	weights := make([]float64, w*h)

	bMin := b.Bounds().Min

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				ca := color.NRGBAModel.Convert(a.At(bounds.Min.X+x, bounds.Min.Y+y)).(color.NRGBA)
				cb := color.NRGBAModel.Convert(b.At(bMin.X+x, bMin.Y+y)).(color.NRGBA)

				d := DeltaE(ca, cb)
				if d < 0 || math.IsNaN(d) {
					d = 0
				}

				idx := y*w + x
				values[idx] = d
				weights[idx] = (float64(ca.A) + float64(cb.A)) / (2 * 255)

				intensity := uint8(math.Round(math.Min(d/scale, 1) * 255))
				pos := y*heatmap.Stride + x*4
				heatmap.Pix[pos+0] = intensity
				heatmap.Pix[pos+1] = intensity
				heatmap.Pix[pos+2] = intensity
				heatmap.Pix[pos+3] = ca.A
			}
		}
	})

	var weightSum, weightedSum, max float64
	for i, d := range values {
		weight := weights[i]
		if weight == 0 {
			continue
		}
		weightSum += weight
		weightedSum += weight * d
		if d > max {
			max = d
		}
	}
	if weightSum > 0 {
		result.Mean = weightedSum / weightSum
	}
	result.Max = max
	result.Heatmap = heatmap
	result.Values = values

	return result, nil
}

// DeltaE returns the CIEDE2000 color difference between two colors, which are
// interpreted as sRGB and converted to CIE L*a*b* under the D65 illuminant.
// The result is always >= 0; identical colors return exactly 0.
func DeltaE(c1, c2 color.Color) float64 {
	l1 := srgbToLab(color.NRGBAModel.Convert(c1).(color.NRGBA))
	l2 := srgbToLab(color.NRGBAModel.Convert(c2).(color.NRGBA))
	return deltaE2000(l1, l2)
}

// labColor holds a color in the CIE L*a*b* color space (D65 illuminant).
type labColor struct {
	l, a, b float64
}

// srgbToLab converts a non-premultiplied sRGB color to CIE L*a*b* (D65).
func srgbToLab(c color.NRGBA) labColor {
	// sRGB to linear RGB
	linearize := func(v float64) float64 {
		if v <= 0.04045 {
			return v / 12.92
		}
		return math.Pow((v+0.055)/1.055, 2.4)
	}
	r := linearize(float64(c.R) / 255)
	g := linearize(float64(c.G) / 255)
	b := linearize(float64(c.B) / 255)

	// Linear sRGB to XYZ (D65)
	x := 0.4124564*r + 0.3575761*g + 0.1804375*b
	y := 0.2126729*r + 0.7151522*g + 0.0721750*b
	z := 0.0193339*r + 0.1191920*g + 0.9503041*b

	// XYZ to L*a*b* with the D65 white point
	const (
		whiteX = 0.95047
		whiteY = 1.0
		whiteZ = 1.08883
	)
	f := func(t float64) float64 {
		const (
			delta  = 6.0 / 29.0
			delta3 = delta * delta * delta
		)
		if t > delta3 {
			return math.Cbrt(t)
		}
		return t/(3*delta*delta) + 4.0/29.0
	}
	fx := f(x / whiteX)
	fy := f(y / whiteY)
	fz := f(z / whiteZ)

	return labColor{
		l: 116*fy - 16,
		a: 500 * (fx - fy),
		b: 200 * (fy - fz),
	}
}

// deltaE2000 returns the CIEDE2000 difference between two L*a*b* colors using
// the standard parametric weights kL = kC = kH = 1.
func deltaE2000(c1, c2 labColor) float64 {
	const (
		degToRad = math.Pi / 180
		radToDeg = 180 / math.Pi
	)

	pow7 := func(v float64) float64 {
		v2 := v * v
		v4 := v2 * v2
		return v4 * v2 * v
	}

	// Hue angle in degrees, 0 when the color has no chroma.
	hue := func(a, b float64) float64 {
		if a == 0 && b == 0 {
			return 0
		}
		h := math.Atan2(b, a) * radToDeg
		if h < 0 {
			h += 360
		}
		return h
	}

	avgL := (c1.l + c2.l) / 2
	chroma1 := math.Hypot(c1.a, c1.b)
	chroma2 := math.Hypot(c2.a, c2.b)
	avgChroma := (chroma1 + chroma2) / 2

	avgChroma7 := pow7(avgChroma)
	g := 0.5 * (1 - math.Sqrt(avgChroma7/(avgChroma7+pow7(25))))

	a1 := c1.a * (1 + g)
	a2 := c2.a * (1 + g)
	c1p := math.Hypot(a1, c1.b)
	c2p := math.Hypot(a2, c2.b)
	avgCp := (c1p + c2p) / 2

	h1 := hue(a1, c1.b)
	h2 := hue(a2, c2.b)

	// hueEps absorbs floating-point noise for hue differences landing
	// exactly on the 180-degree branch boundary (a known CIEDE2000
	// discontinuity), so results match the reference implementation.
	const hueEps = 1e-9
	absHueDiff := math.Abs(h1 - h2)

	var avgH float64
	switch {
	case c1p*c2p == 0:
		avgH = h1 + h2
	case absHueDiff <= 180+hueEps:
		avgH = (h1 + h2) / 2
	case h1+h2 < 360:
		avgH = (h1 + h2 + 360) / 2
	default:
		avgH = (h1 + h2 - 360) / 2
	}

	t := 1 -
		0.17*math.Cos((avgH-30)*degToRad) +
		0.24*math.Cos(2*avgH*degToRad) +
		0.32*math.Cos((3*avgH+6)*degToRad) -
		0.20*math.Cos((4*avgH-63)*degToRad)

	dL := c2.l - c1.l
	dC := c2p - c1p

	var dh float64
	switch {
	case c1p*c2p == 0:
		dh = 0
	case absHueDiff <= 180+hueEps:
		dh = h2 - h1
	case h2-h1 > 180:
		dh = h2 - h1 - 360
	default:
		dh = h2 - h1 + 360
	}
	dH := 2 * math.Sqrt(c1p*c2p) * math.Sin(dh/2*degToRad)

	dL50 := avgL - 50
	sL := 1 + 0.015*dL50*dL50/math.Sqrt(20+dL50*dL50)
	sC := 1 + 0.045*avgCp
	sH := 1 + 0.015*avgCp*t

	deltaTheta := 30 * math.Exp(-math.Pow((avgH-275)/25, 2))
	avgCp7 := pow7(avgCp)
	rC := 2 * math.Sqrt(avgCp7/(avgCp7+pow7(25)))
	rT := -rC * math.Sin(2*deltaTheta*degToRad)

	vL := dL / sL
	vC := dC / sC
	vH := dH / sH

	sum := vL*vL + vC*vC + vH*vH + rT*vC*vH
	if sum <= 0 {
		// The rotation term can push the sum slightly below zero for
		// near-identical colors; the difference is zero in that case.
		return 0
	}
	return math.Sqrt(sum)
}
