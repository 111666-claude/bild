package blur

import (
	"errors"
	"fmt"
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
)

var (
	// ErrNilImage is returned when a nil image is passed to Bilateral.
	ErrNilImage = errors.New("blur: bilateral source image is nil")
	// ErrInvalidRadius is returned when the bilateral radius is not finite or not larger than zero.
	ErrInvalidRadius = errors.New("blur: bilateral radius must be finite and greater than zero")
	// ErrInvalidSpatialSigma is returned when the spatial sigma is not finite or not larger than zero.
	ErrInvalidSpatialSigma = errors.New("blur: bilateral spatial sigma must be finite and greater than zero")
	// ErrInvalidColorSigma is returned when the color sigma is not finite or not larger than zero.
	ErrInvalidColorSigma = errors.New("blur: bilateral color sigma must be finite and greater than zero")
)

// BilateralOptions are the parameters for BilateralFilter.
type BilateralOptions struct {
	// Radius is the size of the neighborhood in pixels. Must be larger than zero.
	Radius float64
	// SpatialSigma controls how fast the weight decays with distance. Must be larger than zero.
	SpatialSigma float64
	// ColorSigma controls how fast the weight decays with color difference, in the 0 to 255 range.
	// Must be larger than zero.
	ColorSigma float64
}

// BilateralFilter is the BilateralOptions flavour of Bilateral.
//
// Usage example:
//
//	result, err := blur.BilateralFilter(img, blur.BilateralOptions{
//		Radius:       3,
//		SpatialSigma: 3,
//		ColorSigma:   25,
//	})
func BilateralFilter(src image.Image, o BilateralOptions) (*image.RGBA, error) {
	return Bilateral(src, o.Radius, o.SpatialSigma, o.ColorSigma)
}

// Bilateral returns an edge-preserving blurred version of the image.
//
// Each output pixel is a weighted average of its neighborhood, where a
// neighbor's weight is the product of a spatial Gaussian (distance to the
// pixel) and a photometric Gaussian (difference to the pixel's color). Flat
// regions are smoothed, but large color differences are down-weighted, so
// sharp edges such as character outlines are preserved.
//
// Alpha is treated as a fourth color channel: it participates in the
// photometric distance and is averaged with exactly the same weights as the
// color channels. The computation stays in premultiplied RGBA, which means
// fully transparent pixels contribute a zero premultiplied color and the
// result remains a valid premultiplied image. Partially transparent edge
// pixels therefore keep their un-premultiplied hue instead of developing a
// dark or colored fringe. Colors are converted to RGBA with image/draw first,
// so non-premultiplied (NRGBA/NRGBA64), grayscale and 16-bit inputs are
// handled correctly.
//
// Pixels outside the image bounds are taken from the closest edge pixel,
// matching the edge extension used by the other blur functions. The output
// has the same bounds as src and src is never modified.
//
// Radius, spatialSigma and colorSigma must all be finite and larger than
// zero; otherwise a wrapped ErrInvalidRadius, ErrInvalidSpatialSigma or
// ErrInvalidColorSigma is returned. A nil image returns ErrNilImage.
//
// The effective neighborhood is capped at ceil(3*spatialSigma) pixels per side,
// because the spatial Gaussian contributes less than 1e-4 of the center weight
// beyond three sigma, and additionally at the largest image dimension, since a
// larger neighborhood only repeats edge pixels. The work and memory therefore
// stay bounded for any finite radius.
//
// The filter performs O(W*H*(2r+1)^2) work and uses O(W*H + r) memory on top of
// its small, fixed-size weight tables (four 256-entry color tables and one
// r+1-entry spatial table). In particular, it does not cache per-pixel
// neighborhoods, so memory does not grow quadratically with the image size or
// with the radius.
//
// Usage example:
//
//	result, err := blur.Bilateral(img, 3, 3, 25)
func Bilateral(src image.Image, radius, spatialSigma, colorSigma float64) (*image.RGBA, error) {
	if src == nil {
		return nil, ErrNilImage
	}
	if math.IsNaN(radius) || math.IsInf(radius, 0) || radius <= 0 {
		return nil, fmt.Errorf("%w: got %v", ErrInvalidRadius, radius)
	}
	if math.IsNaN(spatialSigma) || math.IsInf(spatialSigma, 0) || spatialSigma <= 0 {
		return nil, fmt.Errorf("%w: got %v", ErrInvalidSpatialSigma, spatialSigma)
	}
	if math.IsNaN(colorSigma) || math.IsInf(colorSigma, 0) || colorSigma <= 0 {
		return nil, fmt.Errorf("%w: got %v", ErrInvalidColorSigma, colorSigma)
	}

	bounds := src.Bounds()
	if bounds.Empty() {
		return image.NewRGBA(bounds), nil
	}

	width, height := bounds.Dx(), bounds.Dy()
	minX, minY := bounds.Min.X, bounds.Min.Y

	// Tails beyond three sigma are below 1e-4 of the center weight and can be
	// dropped without a visible difference.
	if radius > 3*spatialSigma {
		radius = 3 * spatialSigma
	}
	// A neighborhood larger than the image only repeats edge pixels, so cap it.
	maxDim := width
	if height > maxDim {
		maxDim = height
	}
	if radius > float64(maxDim) {
		radius = float64(maxDim)
	}
	r := int(math.Ceil(radius))

	// The spatial Gaussian is separable per axis, so a single 1D table of
	// distances serves both dx and dy.
	spatialWeights := make([]float64, r+1)
	spatialDenom := 2 * spatialSigma * spatialSigma
	if spatialDenom == 0 {
		// The sigma squared underflowed to zero: only the center contributes.
		spatialWeights[0] = 1
	} else {
		for i := range spatialWeights {
			fi := float64(i)
			spatialWeights[i] = math.Exp(-(fi * fi) / spatialDenom)
		}
	}

	// The RGBA difference Gaussian is separable per channel, so four small
	// 8-bit-indexed tables replace a 4*255^2-entry lookup table.
	colorWeights := make([][256]float64, 4)
	colorDenom := 2 * colorSigma * colorSigma
	if colorDenom == 0 {
		// The sigma squared underflowed to zero: only an exact match contributes.
		for c := range colorWeights {
			colorWeights[c][0] = 1
		}
	} else {
		for c := range colorWeights {
			for i := 0; i < 256; i++ {
				fi := float64(i)
				colorWeights[c][i] = math.Exp(-(fi * fi) / colorDenom)
			}
		}
	}

	// Reading a shallow reference is safe: the source is only read below.
	srcRGBA := clone.AsShallowRGBA(src)
	srcPix := srcRGBA.Pix

	dst := image.NewRGBA(bounds)
	dstPix := dst.Pix

	parallel.Line(height, func(start, end int) {
		for y := start; y < end; y++ {
			dstRow := dst.PixOffset(minX, minY+y)

			for x := 0; x < width; x++ {
				centerPos := srcRGBA.PixOffset(minX+x, minY+y)
				cr := int(srcPix[centerPos+0])
				cg := int(srcPix[centerPos+1])
				cb := int(srcPix[centerPos+2])
				ca := int(srcPix[centerPos+3])

				var sumR, sumG, sumB, sumA, sumW float64

				for dy := -r; dy <= r; dy++ {
					sy := y + dy
					if sy < 0 {
						sy = 0
					} else if sy >= height {
						sy = height - 1
					}

					rowWeight := spatialWeights[absInt(dy)]
					srcRow := srcRGBA.PixOffset(minX, minY+sy)

					for dx := -r; dx <= r; dx++ {
						sx := x + dx
						if sx < 0 {
							sx = 0
						} else if sx >= width {
							sx = width - 1
						}

						pos := srcRow + sx*4
						pr := int(srcPix[pos+0])
						pg := int(srcPix[pos+1])
						pb := int(srcPix[pos+2])
						pa := int(srcPix[pos+3])

						dr := pr - cr
						dg := pg - cg
						db := pb - cb
						da := pa - ca

						// The photometric Gaussian is separable per channel:
						// exp(-(dr²+dg²+db²+da²)/2σ²) is the product of the
						// per-channel weights.
						colorWeight := colorWeights[0][absInt(dr)] *
							colorWeights[1][absInt(dg)] *
							colorWeights[2][absInt(db)] *
							colorWeights[3][absInt(da)]

						w := rowWeight * spatialWeights[absInt(dx)] * colorWeight
						sumR += w * float64(pr)
						sumG += w * float64(pg)
						sumB += w * float64(pb)
						sumA += w * float64(pa)
						sumW += w
					}
				}

				// sumW is at least the center weight (1), so it cannot be zero.
				invW := 1 / sumW
				pos := dstRow + x*4
				dstPix[pos+0] = clampRound(sumR * invW)
				dstPix[pos+1] = clampRound(sumG * invW)
				dstPix[pos+2] = clampRound(sumB * invW)
				dstPix[pos+3] = clampRound(sumA * invW)
			}
		}
	})

	return dst, nil
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func clampRound(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v + 0.5)
}
