package histogram

import (
	"image"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
)

// luminanceBins is the number of luminance levels in an 8-bit image.
const luminanceBins = 256

// Luminance weights. They are kept as integers so that the histogram and the
// mapping use exactly the same value: 0.3R + 0.6G + 0.1B == (3R + 6G + B) / 10.
const (
	luminanceR     = 3
	luminanceG     = 6
	luminanceB     = 1
	luminanceScale = luminanceR + luminanceG + luminanceB
)

// NewLuminanceHistogram constructs a 256-bin histogram of the perceived
// luminance of the image's pixels.
//
// Luminance is computed with the same 0.3R + 0.6G + 0.1B heuristic used by
// effect.Grayscale and rounded to the nearest of the 256 bins. Fully
// transparent pixels (alpha == 0) carry no visible color and are excluded from
// the counts. Partially transparent pixels are counted; their alpha value does
// not affect their luminance.
func NewLuminanceHistogram(img image.Image) *Histogram {
	h := Histogram{Bins: make([]int, luminanceBins)}
	if img == nil {
		return &h
	}
	accumulateLuminance(h.Bins, clone.AsShallowRGBA(img))
	return &h
}

// LuminanceLUT returns a monotonic 256-entry lookup table that maps every
// source luminance level to a target level taken from the reference image's
// cumulative luminance distribution.
//
// The table is built by matching the normalized cumulative histograms: each
// source level is sampled at the midpoint of its cumulative interval and maps
// to the smallest reference level whose cumulative share reaches that point.
// For a populated source bin this is the usual quantile mapping; for a flat or
// sparse distribution it avoids collapsing every pixel onto an extreme of the
// reference. Because both cumulative distributions are non-decreasing, the
// resulting table is non-decreasing as well.
//
// Fully transparent pixels are ignored in both distributions. If either image
// is nil or has no visible pixels, the identity mapping is returned, so callers
// can use the table unconditionally.
func LuminanceLUT(src, ref image.Image) []uint8 {
	if src == nil || ref == nil {
		return identityLUT()
	}
	srcHist := NewLuminanceHistogram(src)
	refHist := NewLuminanceHistogram(ref)
	return buildLuminanceLUT(srcHist.Bins, refHist.Bins)
}

// MatchLuminance returns a copy of src whose luminance distribution is matched
// to the reference image using their cumulative luminance histograms.
//
// The returned image keeps the source dimensions and bounds, hue, and alpha
// values. Fully transparent source pixels are copied unchanged and do not
// contribute to either distribution. The input images are never modified.
//
// Every visible pixel is mapped through the monotonic 256-level lookup table
// returned by LuminanceLUT and then scaled towards black or white, which
// preserves its hue while reaching the target luminance. The work is linear in
// the number of source pixels, and the extra memory is O(1) beyond the copy of
// the image.
//
// Degenerate inputs are handled without divisions by zero: an empty or fully
// transparent source is returned as a copy, and a nil or fully transparent
// reference leaves the source unchanged. A single-color source maps to a single
// target level, so uniform areas remain uniform.
func MatchLuminance(src, ref image.Image) *image.RGBA {
	if src == nil {
		return &image.RGBA{}
	}

	dst := clone.AsRGBA(src)
	if ref == nil {
		return dst
	}

	srcBins := make([]int, luminanceBins)
	srcCount := accumulateLuminance(srcBins, dst)

	refBins := make([]int, luminanceBins)
	refCount := accumulateLuminance(refBins, clone.AsShallowRGBA(ref))

	// Nothing to match (or nothing to match against): leave the source as is.
	if srcCount == 0 || refCount == 0 {
		return dst
	}

	lut := buildLuminanceLUT(srcBins, refBins)

	bounds := dst.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			row := y * dst.Stride
			for x := 0; x < w; x++ {
				pos := row + x*4

				// Transparent pixels have no visible luminance and are left
				// untouched, including their RGB components.
				if dst.Pix[pos+3] == 0 {
					continue
				}

				r := int(dst.Pix[pos+0])
				g := int(dst.Pix[pos+1])
				b := int(dst.Pix[pos+2])
				lum := luminanceR*r + luminanceG*g + luminanceB*b

				// Map the pixel's luminance bin through the table. Applying the
				// table at bin resolution (rather than interpolating between
				// entries) keeps the output histogram an exact remap of the
				// source histogram, which is what makes the distributions match.
				target := lookupLuminance(lut, lum)
				if target == lum {
					continue
				}

				if target < lum {
					// Darken towards black: scaling all channels by the same
					// factor keeps their ratios and therefore the hue.
					dst.Pix[pos+0] = uint8(divRound(r*target, lum))
					dst.Pix[pos+1] = uint8(divRound(g*target, lum))
					dst.Pix[pos+2] = uint8(divRound(b*target, lum))
					continue
				}

				// Lighten towards white: adding the same offset to all channels
				// also keeps the hue. lum < 2550 because target <= 2550.
				den := luminanceScale*255 - lum
				delta := target - lum
				dst.Pix[pos+0] = uint8(r + divRound((255-r)*delta, den))
				dst.Pix[pos+1] = uint8(g + divRound((255-g)*delta, den))
				dst.Pix[pos+2] = uint8(b + divRound((255-b)*delta, den))
			}
		}
	})

	return dst
}

// accumulateLuminance counts the visible pixels of src per luminance bin and
// returns the number of pixels it counted.
func accumulateLuminance(bins []int, src *image.RGBA) int {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	count := 0

	for y := 0; y < h; y++ {
		row := y * src.Stride
		for x := 0; x < w; x++ {
			pos := row + x*4
			if src.Pix[pos+3] == 0 {
				continue
			}
			lum := luminanceR*int(src.Pix[pos+0]) +
				luminanceG*int(src.Pix[pos+1]) +
				luminanceB*int(src.Pix[pos+2])
			bins[(lum+luminanceScale/2)/luminanceScale]++
			count++
		}
	}

	return count
}

// buildLuminanceLUT returns the monotonic mapping described by LuminanceLUT.
// The two slices hold the non-cumulative luminance histograms of the source
// and reference images.
func buildLuminanceLUT(srcBins, refBins []int) []uint8 {
	lut := identityLUT()
	if len(srcBins) != luminanceBins || len(refBins) != luminanceBins {
		return lut
	}

	srcCDF := (&Histogram{Bins: srcBins}).Cumulative().Bins
	refCDF := (&Histogram{Bins: refBins}).Cumulative().Bins

	srcTotal := srcCDF[luminanceBins-1]
	refTotal := refCDF[luminanceBins-1]
	if srcTotal <= 0 || refTotal <= 0 {
		return lut
	}

	// Walk both cumulative distributions once. Each source bin is sampled at
	// the midpoint of its cumulative interval: for a populated bin this is the
	// textbook quantile mapping, while for a flat or sparse distribution it
	// avoids collapsing every pixel onto an extreme of the reference. j only
	// moves forward, which keeps the table monotonic and the construction
	// O(256).
	stride := 2 * float64(srcTotal)
	j := 0
	for i := 0; i < luminanceBins; i++ {
		// (cumulative before bin i) + half of bin i, doubled to stay integral.
		share := float64(2*(srcCDF[i]-srcBins[i])+srcBins[i]) / stride
		for j < luminanceBins-1 && float64(refCDF[j])/float64(refTotal) < share {
			j++
		}
		lut[i] = uint8(j)
	}

	return lut
}

// lookupLuminance returns the mapped luminance, in the 0 to 2550 range, for a
// weighted source luminance. The value is read from the table entry of the
// corresponding luminance bin.
func lookupLuminance(lut []uint8, lum int) int {
	// lum is a sum of three channel weights times 8-bit values, so it lies in
	// [0, 255*luminanceScale] and its bin always indexes the table.
	bin := (lum + luminanceScale/2) / luminanceScale
	return int(lut[bin]) * luminanceScale
}

// identityLUT returns a lookup table that maps every level to itself.
func identityLUT() []uint8 {
	lut := make([]uint8, luminanceBins)
	for i := range lut {
		lut[i] = uint8(i)
	}
	return lut
}

// divRound divides two non-negative integers and rounds to the nearest one.
func divRound(numerator, denominator int) int {
	return (numerator + denominator/2) / denominator
}
