package blur

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/math/f64"
	"github.com/anthonynsimon/bild/parallel"
)

// Bilateral applies a bilateral filter to the image and returns the result.
//
// The bilateral filter smooths flat regions while preserving hard edges: every
// output pixel is a weighted average of its neighbors in which weights combine
// a spatial (geometric distance) Gaussian and a color (photometric distance)
// Gaussian. Neighbors that are far away in space or whose color differs
// strongly from the center pixel receive a negligible weight, so edges are not
// smeared as they would be by a plain Gaussian blur.
//
// Parameters:
//   - radius is the spatial support radius in pixels. The search window is a
//     square with side length 2*floor(radius)+1, centered on each pixel.
//   - spatialSigma is the standard deviation in pixels of the spatial Gaussian.
//   - colorSigma is the standard deviation, expressed in 0-255 RGB channel
//     units, of the color Gaussian. Smaller values preserve edges more
//     aggressively; larger values make the filter behave closer to a plain
//     Gaussian blur.
//
// The color distance is the Euclidean distance between RGB channels on the
// 0-255 scale; the alpha channel does not participate in it. Each neighbor's
// weight is the product of its spatial weight, its color weight and its
// normalized alpha (0..1). Fully transparent neighbors therefore contribute
// nothing and cannot pollute visible colors. The output alpha is copied
// unchanged from the source pixel.
//
// Sampling outside the image bounds uses edge extension (the closest border
// pixel is repeated). The returned image is always a new *image.RGBA with the
// same bounds as the source, regardless of the source color model.
//
// Deterministic handling of degenerate parameters:
//   - radius <= 0, NaN or +Inf, or spatialSigma <= 0 or NaN, or
//     colorSigma <= 0 or NaN: the image is returned unchanged as an RGBA copy.
//   - A radius larger than the image's largest dimension is clamped to it.
//   - Empty images return an empty *image.RGBA.
//   - If no neighbor carries any weight (e.g. an opaque pixel surrounded only
//     by fully transparent pixels), the source pixel color is kept.
func Bilateral(src image.Image, radius, spatialSigma, colorSigma float64) *image.RGBA {
	// Degenerate parameters: return an unchanged RGBA copy, matching the
	// convention used by the other blur functions for non-positive radius.
	if math.IsNaN(radius) || math.IsNaN(spatialSigma) || math.IsNaN(colorSigma) {
		return clone.AsRGBA(src)
	}
	if radius <= 0 || math.IsInf(radius, 1) ||
		spatialSigma <= 0 || math.IsInf(spatialSigma, 0) ||
		colorSigma <= 0 || math.IsInf(colorSigma, 0) {
		return clone.AsRGBA(src)
	}

	bounds := src.Bounds()
	if bounds.Empty() {
		return clone.AsRGBA(src)
	}

	w, h := bounds.Dx(), bounds.Dy()

	// Clamp the radius to the largest image dimension. A window may never need
	// to reach further than that, and this prevents pathological allocations
	// (and float-to-int overflow) on absurdly large radii.
	r := int(math.Floor(radius))
	maxDim := w
	if h > maxDim {
		maxDim = h
	}
	if r > maxDim {
		r = maxDim
	}
	if r <= 0 {
		return clone.AsRGBA(src)
	}

	// Edge-extended padding means every window sample is always in bounds,
	// including for pixels on the image border.
	padded := clone.Pad(src, r, r, clone.EdgeExtend)
	padStride := padded.Stride

	// Pre-compute the separable spatial Gaussian kernel.
	kernelLen := 2*r + 1
	spatial := make([]float64, kernelLen)
	invTwoSpatialSigmaSq := 1.0 / (2.0 * spatialSigma * spatialSigma)
	for i := 0; i < kernelLen; i++ {
		d := float64(i - r)
		spatial[i] = math.Exp(-(d * d) * invTwoSpatialSigmaSq)
	}
	invTwoColorSigmaSq := 1.0 / (2.0 * colorSigma * colorSigma)

	dst := image.NewRGBA(bounds)

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				centerPos := (y+r)*padStride + (x+r)*4
				centerR := float64(padded.Pix[centerPos+0])
				centerG := float64(padded.Pix[centerPos+1])
				centerB := float64(padded.Pix[centerPos+2])
				centerA := padded.Pix[centerPos+3]

				dstPos := y*dst.Stride + x*4

				// Fully transparent center: there is no visible color to
				// filter, so pass the pixel through untouched.
				if centerA == 0 {
					dst.Pix[dstPos+0] = padded.Pix[centerPos+0]
					dst.Pix[dstPos+1] = padded.Pix[centerPos+1]
					dst.Pix[dstPos+2] = padded.Pix[centerPos+2]
					dst.Pix[dstPos+3] = 0
					continue
				}

				var sumR, sumG, sumB, sumW float64
				for ky := -r; ky <= r; ky++ {
					row := (y + r + ky) * padStride
					wy := spatial[ky+r]
					for kx := -r; kx <= r; kx++ {
						pos := row + (x+r+kx)*4

						alphaWeight := float64(padded.Pix[pos+3]) / 255.0
						if alphaWeight == 0 {
							// Transparent neighbor: skip so it cannot pollute
							// the visible color.
							continue
						}

						nR := float64(padded.Pix[pos+0])
						nG := float64(padded.Pix[pos+1])
						nB := float64(padded.Pix[pos+2])

						dR := nR - centerR
						dG := nG - centerG
						dB := nB - centerB
						colorDistSq := dR*dR + dG*dG + dB*dB

						weight := wy * spatial[kx+r] *
							math.Exp(-colorDistSq*invTwoColorSigmaSq) * alphaWeight

						sumR += nR * weight
						sumG += nG * weight
						sumB += nB * weight
						sumW += weight
					}
				}

				if sumW == 0 {
					// No weighted neighbor (opaque center surrounded by
					// transparent pixels): keep the source color.
					dst.Pix[dstPos+0] = padded.Pix[centerPos+0]
					dst.Pix[dstPos+1] = padded.Pix[centerPos+1]
					dst.Pix[dstPos+2] = padded.Pix[centerPos+2]
				} else {
					dst.Pix[dstPos+0] = uint8(f64.Clamp(sumR/sumW+0.5, 0, 255))
					dst.Pix[dstPos+1] = uint8(f64.Clamp(sumG/sumW+0.5, 0, 255))
					dst.Pix[dstPos+2] = uint8(f64.Clamp(sumB/sumW+0.5, 0, 255))
				}
				// Alpha is preserved from the source pixel.
				dst.Pix[dstPos+3] = centerA
			}
		}
	})

	return dst
}
