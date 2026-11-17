package transform

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/math/f64"
	"github.com/anthonynsimon/bild/parallel"
)

// normalMapZeroLength is the decoded vector length below which a normal is treated as
// zero-length. Tangent-space normals are expected to be unit length, so shorter vectors
// carry no usable direction. The threshold also covers the canonical 8-bit zero encoding
// (128, 128, 128), whose decoded length is ~0.0068 rather than exactly zero.
const normalMapZeroLength = 1e-2

// maxImagePixels is the largest pixel count that image.NewRGBA can allocate and address
// without overflowing the pixel buffer size calculation.
const maxImagePixels = (math.MaxInt - 1) / 4

// ResizeNormalMap returns a new image with its size adjusted to the new width and height,
// treating the source pixels as tangent-space normals encoded in RGB (each channel maps
// 0..255 to -1..1) instead of as colors.
//
// The filter parameter is the same ResampleFilter used by Resize: its support and weights
// select and weight the source samples in the same way. Unlike Resize, the decoded normals
// are interpolated as vectors, the result is renormalized to unit length, and then encoded
// back into RGB. The alpha channel is interpolated with the same filter, so transparency is
// preserved.
//
// Samples are weighted by their alpha, so fully transparent pixels do not contribute a
// direction. Samples whose filter weight is not finite are skipped. If the accumulated
// weight is not finite or sums to zero, if an accumulated value overflows, or if every
// contributing sample is transparent, the pixel falls back to the nearest source sample,
// with its direction renormalized and its alpha copied. If the alpha weighted vectors
// cancel out or are shorter than the zero-length threshold, the pixel is encoded as the
// flat tangent-space normal (128, 128, 255), so every output pixel carries a unit-length
// vector.
//
// Invalid target dimensions (width or height less than 1, or a size whose pixel buffer
// cannot be addressed), a nil image, and an empty source image return an empty image,
// matching Resize. A filter with nil Fn, a non-positive support, or non-finite support
// cannot be evaluated and falls back to nearest neighbor sampling. The source image is
// never modified.
//
// Non-zero image bounds are supported: samples are read relative to the bounds origin and
// the returned image has its origin at (0, 0), the same as Resize.
//
// Usage example:
//
//	result := transform.ResizeNormalMap(normalMap, 1024, 1024, transform.Linear)
func ResizeNormalMap(img image.Image, width, height int, filter ResampleFilter) *image.RGBA {
	if img == nil || width <= 0 || height <= 0 || img.Bounds().Empty() ||
		width > maxImagePixels/height {
		return image.NewRGBA(image.Rect(0, 0, 0, 0))
	}

	src := clone.AsShallowRGBA(img)

	// Nearest neighbor is a special case, and a filter that cannot be evaluated is
	// handled by the nearest sample fallback.
	if filter.Support <= 0 || filter.Fn == nil ||
		math.IsNaN(filter.Support) || math.IsInf(filter.Support, 0) {
		return resampleNormalMapNearest(src, width, height)
	}

	return resampleNormalMap(src, width, height, filter)
}

// resampleNormalMap performs a two-dimensional weighted interpolation of the decoded
// normals and renormalizes once at the end. Working in a single pass keeps the alpha
// weighting exact and avoids quantizing an intermediate image.
func resampleNormalMap(src *image.RGBA, width, height int, filter ResampleFilter) *image.RGBA {
	srcBounds := src.Bounds()
	srcW, srcH := srcBounds.Dx(), srcBounds.Dy()

	deltaX := float64(srcW) / float64(width)
	deltaY := float64(srcH) / float64(height)
	// Scale must be at least 1. Special case for image size reduction filter radius.
	scaleX := math.Max(deltaX, 1.0)
	scaleY := math.Max(deltaY, 1.0)
	radiusX := math.Ceil(scaleX * filter.Support)
	radiusY := math.Ceil(scaleY * filter.Support)
	// Guard the sample window against overflow for pathological support values.
	if radiusX > float64(srcW) {
		radiusX = float64(srcW)
	}
	if radiusY > float64(srcH) {
		radiusY = float64(srcH)
	}

	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	dstStride := dst.Stride

	parallel.Line(height, func(start, end int) {
		// Weights are precomputed per row and per column so each sample's filter value
		// is evaluated once per destination pixel instead of once per channel.
		yWeights := make([]float64, 0, int(2*radiusY)+2)
		xWeights := make([]float64, 0, int(2*radiusX)+2)

		for y := start; y < end; y++ {
			iy := (float64(y)+0.5)*deltaY - 0.5
			ystart, yend := int(iy-radiusY+0.5), int(iy+radiusY)
			if ystart < 0 {
				ystart = 0
			}
			if yend >= srcH {
				yend = srcH - 1
			}

			yWeights = yWeights[:0]
			for ky := ystart; ky <= yend; ky++ {
				wy := filter.Fn((float64(ky) - iy) / scaleY)
				if !isFiniteFloat(wy) {
					wy = 0
				}
				yWeights = append(yWeights, wy)
			}

			for x := 0; x < width; x++ {
				ix := (float64(x)+0.5)*deltaX - 0.5
				xstart, xend := int(ix-radiusX+0.5), int(ix+radiusX)
				if xstart < 0 {
					xstart = 0
				}
				if xend >= srcW {
					xend = srcW - 1
				}

				xWeights = xWeights[:0]
				for kx := xstart; kx <= xend; kx++ {
					wx := filter.Fn((float64(kx) - ix) / scaleX)
					if !isFiniteFloat(wx) {
						wx = 0
					}
					xWeights = append(xWeights, wx)
				}

				// sumW is the raw filter weight sum used for alpha interpolation,
				// sumA the alpha weighted sum, and absA the sum of the absolute
				// alpha weighted samples. absA is always non-negative and scales
				// with the total sample energy, so dividing the vector sums by it
				// makes the zero-length test independent of the filter's
				// normalization and of the number of samples. vx/vy/vz accumulate
				// the decoded normals weighted by both the filter and the sample
				// alpha.
				var sumW, sumA, absA, vx, vy, vz float64

				for ky := ystart; ky <= yend; ky++ {
					wy := yWeights[ky-ystart]
					// The row contribution can be skipped entirely when the row
					// weight is zero.
					if wy == 0 {
						continue
					}

					rowPos := src.PixOffset(srcBounds.Min.X+xstart, srcBounds.Min.Y+ky)
					for i, wx := range xWeights {
						w := wx * wy
						if w == 0 {
							continue
						}

						pos := rowPos + i*4
						sumW += w

						a := float64(src.Pix[pos+3]) / 255.0
						if a == 0 {
							// Transparent pixels have no meaningful direction.
							continue
						}

						sumA += w * a
						absA += math.Abs(w * a)
						vx += w * a * (float64(src.Pix[pos+0])*(2.0/255.0) - 1)
						vy += w * a * (float64(src.Pix[pos+1])*(2.0/255.0) - 1)
						vz += w * a * (float64(src.Pix[pos+2])*(2.0/255.0) - 1)
					}
				}

				dstPos := y*dstStride + x*4

				if !isFiniteFloat(sumW) || !isFiniteFloat(sumA) || !isFiniteFloat(absA) ||
					!isFiniteFloat(vx) || !isFiniteFloat(vy) || !isFiniteFloat(vz) ||
					sumW == 0 || absA == 0 {
					// The filter produced unusable weights, or every contributing
					// sample was transparent. Use a fixed fallback: the nearest
					// source sample, normalized.
					writeNormalSample(dst, dstPos, src, x, y, width, height)
					continue
				}

				alpha := sumA / sumW
				if !isFiniteFloat(alpha) {
					writeNormalSample(dst, dstPos, src, x, y, width, height)
					continue
				}

				// The resolved source vector. The zero-length threshold is relative
				// to absA, the sum of the absolute alpha weighted samples, so it does
				// not depend on the filter's normalization or on the number of
				// samples considered.
				nx, ny, nz := normalizeToFlat(vx, vy, vz, normalMapZeroLength*absA)

				dst.Pix[dstPos+0] = encodeNormalChannel(nx)
				dst.Pix[dstPos+1] = encodeNormalChannel(ny)
				dst.Pix[dstPos+2] = encodeNormalChannel(nz)
				dst.Pix[dstPos+3] = uint8(f64.Clamp(alpha*255+0.5, 0, 255))
			}
		}
	})

	return dst
}

// resampleNormalMapNearest copies the nearest source sample's direction and alpha. The
// decoded sample vector is renormalized, and transparent or zero-length samples are
// encoded as the flat normal.
func resampleNormalMapNearest(src *image.RGBA, width, height int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	dstStride := dst.Stride

	parallel.Line(height, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < width; x++ {
				writeNormalSample(dst, y*dstStride+x*4, src, x, y, width, height)
			}
		}
	})

	return dst
}

// writeNormalSample writes the normalized nearest source sample of the destination pixel
// (x, y) at dstPos. Transparent and zero-length samples are encoded as the flat normal.
func writeNormalSample(dst *image.RGBA, dstPos int, src *image.RGBA, x, y, width, height int) {
	nx, ny, nz, a := nearestNormal(src, x, y, width, height)
	dst.Pix[dstPos+0] = encodeNormalChannel(nx)
	dst.Pix[dstPos+1] = encodeNormalChannel(ny)
	dst.Pix[dstPos+2] = encodeNormalChannel(nz)
	dst.Pix[dstPos+3] = a
}

// nearestNormal returns the normalized direction and alpha of the source sample that the
// nearest neighbor resampling would select for the destination pixel (x, y).
func nearestNormal(src *image.RGBA, x, y, width, height int) (nx, ny, nz float64, a uint8) {
	srcBounds := src.Bounds()
	srcW, srcH := srcBounds.Dx(), srcBounds.Dy()

	sx := int((float64(x) + 0.5) * float64(srcW) / float64(width))
	sy := int((float64(y) + 0.5) * float64(srcH) / float64(height))
	if sx < 0 {
		sx = 0
	} else if sx >= srcW {
		sx = srcW - 1
	}
	if sy < 0 {
		sy = 0
	} else if sy >= srcH {
		sy = srcH - 1
	}

	pos := src.PixOffset(srcBounds.Min.X+sx, srcBounds.Min.Y+sy)
	a = src.Pix[pos+3]
	if a == 0 {
		return 0, 0, 1, 0
	}
	nx, ny, nz = decodeNormal(src.Pix[pos], src.Pix[pos+1], src.Pix[pos+2])
	// A single decoded normal is expected to be unit length; still guard against
	// zero-length vectors stored in the source.
	nx, ny, nz = normalizeToFlat(nx, ny, nz, normalMapZeroLength)
	return nx, ny, nz, a
}

// decodeNormal maps the 0..255 encoding of a tangent-space normal to the -1..1 range.
func decodeNormal(r, g, b uint8) (float64, float64, float64) {
	return float64(r)*(2.0/255.0) - 1,
		float64(g)*(2.0/255.0) - 1,
		float64(b)*(2.0/255.0) - 1
}

// isFiniteFloat reports whether v is neither NaN nor an infinity.
func isFiniteFloat(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// flatNormalDirection returns the fallback unit normal, pointing straight out of the
// tangent-space surface.
func flatNormalDirection() (float64, float64, float64) {
	return 0, 0, 1
}

// normalizeToFlat normalizes the vector and returns the flat normal if the vector is
// too short to carry a direction or cannot be normalized. The length is compared against
// minLength, except when minLength is zero, where the vector only has to be non-zero.
func normalizeToFlat(x, y, z, minLength float64) (float64, float64, float64) {
	length := math.Hypot(math.Hypot(x, y), z)
	if !isFiniteFloat(length) || length <= 0 || length < minLength {
		return flatNormalDirection()
	}
	return x / length, y / length, z / length
}

// encodeNormalChannel maps a component in the -1..1 range to the 0..255 encoding used by
// tangent-space normal maps.
func encodeNormalChannel(v float64) uint8 {
	return uint8(f64.Clamp(v*127.5+127.5+0.5, 0, 255))
}
