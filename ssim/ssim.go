/*
Package ssim provides local structural similarity (SSIM) comparison of images.

SSIM rates a distorted image against a reference by comparing local luminance,
contrast and structure instead of raw pixel differences. That makes it possible
to tell a global brightness change (which mostly affects the luminance term)
apart from lost detail (which lowers the contrast and structure terms).

Compare returns the mean local score and a per-pixel quality heatmap aligned
with the reference's bounds. Pixels that are not visible in both images (alpha
zero in either) carry no weight, so transparent regions are not treated as
damage.

Usage example:

	result, err := ssim.Compare(reference, candidate, nil)
	if err != nil {
		// handle error
	}
	fmt.Println(result.Score)
	imgio.Save("heatmap.png", result.Heatmap, imgio.PNGEncoder())
*/
package ssim

import (
	"errors"
	"image"
	"image/draw"
	"math"
	"reflect"

	"github.com/anthonynsimon/bild/math/f64"
	"github.com/anthonynsimon/bild/parallel"
)

// Default comparison parameters. They follow the common SSIM formulation for
// 8-bit images.
const (
	// DefaultWindowSize is the side length of the square local window.
	DefaultWindowSize = 7
	// DefaultK1 is the default luminance stabilizer constant.
	DefaultK1 = 0.01
	// DefaultK2 is the default contrast/structure stabilizer constant.
	DefaultK2 = 0.03
	// DefaultDynamicRange is the default dynamic range (L) of the luma values.
	DefaultDynamicRange = 255.0
)

// Luma weights used to convert RGB pixels to the luminance plane. They match
// the 0.3R + 0.6G + 0.1B heuristic used elsewhere in bild.
const (
	lumaR = 0.3
	lumaG = 0.6
	lumaB = 0.1
)

// Comparison errors.
var (
	// ErrNilImage is returned when either input image is nil.
	ErrNilImage = errors.New("ssim: both images are required")
	// ErrEmptyImage is returned when either input image has no pixels.
	ErrEmptyImage = errors.New("ssim: images must not be empty")
	// ErrSizeMismatch is returned when the images have different dimensions.
	ErrSizeMismatch = errors.New("ssim: images must have the same width and height")
	// ErrInvalidWindow is returned when the window size is not a positive odd
	// integer.
	ErrInvalidWindow = errors.New("ssim: window size must be a positive odd integer")
	// ErrNonFiniteParameter is returned when K1, K2 or the dynamic range is
	// NaN or infinite.
	ErrNonFiniteParameter = errors.New("ssim: K1, K2 and dynamic range must be finite")
	// ErrInvalidParameter is returned when K1, K2 or the dynamic range is not
	// positive.
	ErrInvalidParameter = errors.New("ssim: K1, K2 and dynamic range must be positive")
)

// Options configures a local SSIM comparison.
//
// A nil *Options passed to Compare selects DefaultOptions. A non-nil Options
// is used as provided and every field must be valid; starting from
// DefaultOptions makes it possible to change a single field.
type Options struct {
	// WindowSize is the side length, in pixels, of the square comparison
	// window. It must be a positive odd integer. A window larger than the
	// image is allowed: neighborhoods are clipped to the image, so the
	// comparison becomes global.
	WindowSize int
	// K1 stabilizes the luminance term. Typical value: 0.01.
	K1 float64
	// K2 stabilizes the contrast and structure terms. Typical value: 0.03.
	K2 float64
	// DynamicRange is the maximum value of the luma plane. It is 255 for
	// 8-bit images.
	DynamicRange float64
}

// Result holds the outcome of a local SSIM comparison.
type Result struct {
	// Score is the mean local SSIM over all windows that contain visible
	// pixels. It lies in the range -1 to 1, where 1 means the images are
	// structurally identical.
	Score float64
	// Luminance is the mean local luminance term, in the range 0 to 1.
	Luminance float64
	// Contrast is the mean local contrast term, in the range 0 to 1.
	Contrast float64
	// Structure is the mean local structure term, in the range -1 to 1.
	Structure float64
	// Heatmap is an 8-bit quality map with the same bounds as the first
	// image. A value of 255 marks a perfect local score and 0 marks a score
	// of zero or below, so damaged areas appear dark. Pixels whose window has
	// no visible weight are 255.
	Heatmap *image.Gray
}

// DefaultOptions returns the options used when Compare receives nil options.
func DefaultOptions() *Options {
	return &Options{
		WindowSize:   DefaultWindowSize,
		K1:           DefaultK1,
		K2:           DefaultK2,
		DynamicRange: DefaultDynamicRange,
	}
}

// Compare computes the local structural similarity between the reference and
// candidate images and returns the mean score together with a quality
// heatmap.
//
// The images must have the same dimensions. They may have different bounds
// origins; pixels are matched by their relative position, and the returned
// heatmap uses the reference's bounds. Black and white, grayscale, paletted
// and color images are all accepted. Both inputs are only read and are never
// modified.
//
// For every pixel, SSIM is evaluated on the square WindowSize x WindowSize
// neighborhood centered on it. Neighborhoods are clipped to the image bounds,
// so pixels at the edges use a smaller but fully in-bounds window; no padding
// or duplicated edge pixels are introduced. The local score combines the
// luminance, contrast and structure terms using the standard constants
// C1 = (K1 * DynamicRange)^2 and C2 = (K2 * DynamicRange)^2.
//
// Luminance is the 0.3R + 0.6G + 0.1B combination already used by bild,
// computed on non-premultiplied colors so alpha never dims the colors being
// compared. Pixels are weighted by the product of their 8-bit alpha values in
// the two images, so a pixel that is fully transparent in either image
// contributes nothing and a partially transparent pixel contributes less.
// Windows with no visible weight are neutral: they count as a local score of
// 1 and are left out of the mean. Fully transparent images therefore compare
// as identical rather than producing a division by zero.
//
// The calculation is linear in the image area for a fixed window size and uses
// O(width * height) extra memory for the converted input images and heatmap,
// plus O(width) work buffers per CPU. Because the stabilizers are strictly
// positive, constant images produce finite scores and heatmap values.
func Compare(reference, candidate image.Image, options *Options) (*Result, error) {
	if isNilImage(reference) || isNilImage(candidate) {
		return nil, ErrNilImage
	}

	opts, err := resolveOptions(options)
	if err != nil {
		return nil, err
	}

	refBounds := reference.Bounds()
	candBounds := candidate.Bounds()
	if refBounds.Empty() || candBounds.Empty() {
		return nil, ErrEmptyImage
	}

	width, height := refBounds.Dx(), refBounds.Dy()
	if width != candBounds.Dx() || height != candBounds.Dy() {
		return nil, ErrSizeMismatch
	}
	ref := asShallowNRGBA(reference)
	cand := asShallowNRGBA(candidate)

	c1 := math.Pow(opts.K1*opts.DynamicRange, 2)
	c2 := math.Pow(opts.K2*opts.DynamicRange, 2)
	// The stabilizers must stay strictly positive and finite; otherwise a
	// constant image could divide by zero. Extreme but finite parameters can
	// still underflow or overflow here, so check the derived values too.
	if !isFinite(c1) || !isFinite(c2) || c1 <= 0 || c2 <= 0 {
		return nil, ErrInvalidParameter
	}
	radius := opts.WindowSize / 2
	// A radius beyond the image extent is equivalent to the full image and
	// clamping it also keeps the sliding indices far from integer overflow.
	radius = min(radius, max(width, height))

	heatmap := image.NewGray(refBounds)

	type rowStats struct {
		count                                 int
		score, luminance, contrast, structure float64
	}
	stats := make([]rowStats, height)

	parallel.Line(height, func(start, end int) {
		sums := newWindowSums(width, radius, ref, cand)

		// Seed the vertical accumulator with the rows of the window centered
		// on the first row of this chunk.
		top := max(0, start-radius)
		bottom := min(height-1, start+radius)
		for y := top; y <= bottom; y++ {
			sums.addRow(y, 1)
		}

		for y := start; y < end; y++ {
			if y > start {
				newTop := max(0, y-radius)
				newBottom := min(height-1, y+radius)
				if newBottom > bottom {
					sums.addRow(newBottom, 1)
				}
				if newTop > top {
					sums.addRow(newTop-1, -1)
				}
				top, bottom = newTop, newBottom
			}

			var row rowStats
			rowStart := y * heatmap.Stride
			for x := 0; x < width; x++ {
				totalWeight := sums.acc[momentWeight][x]
				// Rounding error in the sliding sums can only leave a tiny
				// residue around an exact zero.
				if totalWeight <= weightEpsilon {
					// Nothing visible in this window: it cannot be damaged.
					heatmap.Pix[rowStart+x] = 0xFF
					continue
				}

				invWeight := 1 / totalWeight
				meanRef := sums.acc[momentRef][x] * invWeight
				meanCand := sums.acc[momentCand][x] * invWeight

				varRef := sums.acc[momentRefSq][x]*invWeight - meanRef*meanRef
				varCand := sums.acc[momentCandSq][x]*invWeight - meanCand*meanCand
				// Rounding can only drive a non-negative variance slightly
				// below zero.
				varRef = math.Max(varRef, 0)
				varCand = math.Max(varCand, 0)
				cov := sums.acc[momentRefCand][x]*invWeight - meanRef*meanCand

				luminance := (2*meanRef*meanCand + c1) /
					(meanRef*meanRef + meanCand*meanCand + c1)
				luminance = f64.Clamp(luminance, 0, 1)

				stdRef := math.Sqrt(varRef)
				stdCand := math.Sqrt(varCand)

				// Contrast-structure is kept as one exact ratio for the local
				// score, while the contrast term is also reported separately.
				// Structure is then derived from the two, which keeps the
				// identity luminance * contrast * structure exact for equal
				// images and avoids amplifying square-root rounding.
				contrastStructure := (2*cov + c2) / (varRef + varCand + c2)
				contrastStructure = f64.Clamp(contrastStructure, -1, 1)

				contrast := (2*stdRef*stdCand + c2) / (varRef + varCand + c2)
				contrast = f64.Clamp(contrast, 0, 1)

				structure := 1.0
				if contrast > 0 {
					structure = contrastStructure / contrast
				}
				structure = f64.Clamp(structure, -1, 1)

				localScore := f64.Clamp(luminance*contrastStructure, -1, 1)

				row.count++
				row.score += localScore
				row.luminance += luminance
				row.contrast += contrast
				row.structure += structure

				// Negative local scores mean anti-correlated structure: they
				// are as bad as it gets and map to black.
				heatmap.Pix[rowStart+x] = heatValue(localScore)
			}
			stats[y] = row
		}
	})

	result := &Result{Heatmap: heatmap}

	var total int
	for _, row := range stats {
		total += row.count
		result.Score += row.score
		result.Luminance += row.luminance
		result.Contrast += row.contrast
		result.Structure += row.structure
	}

	if total == 0 {
		// Nothing visible to compare. Two equally invisible images are not
		// damaged, and returning 1 avoids a NaN.
		result.Score = 1
		result.Luminance = 1
		result.Contrast = 1
		result.Structure = 1
		return result, nil
	}

	invTotal := 1 / float64(total)
	result.Score = f64.Clamp(result.Score*invTotal, -1, 1)
	result.Luminance = f64.Clamp(result.Luminance*invTotal, 0, 1)
	result.Contrast = f64.Clamp(result.Contrast*invTotal, 0, 1)
	result.Structure = f64.Clamp(result.Structure*invTotal, -1, 1)

	return result, nil
}

// Score returns the mean local SSIM of the two images using the default
// options. It is a shortcut for Compare when only the score is needed.
func Score(reference, candidate image.Image) (float64, error) {
	result, err := Compare(reference, candidate, nil)
	if err != nil {
		return 0, err
	}
	return result.Score, nil
}

func resolveOptions(options *Options) (Options, error) {
	if options == nil {
		return *DefaultOptions(), nil
	}

	if options.WindowSize <= 0 || options.WindowSize%2 == 0 {
		return Options{}, ErrInvalidWindow
	}
	if !isFinite(options.K1) || !isFinite(options.K2) || !isFinite(options.DynamicRange) {
		return Options{}, ErrNonFiniteParameter
	}
	if options.K1 <= 0 || options.K2 <= 0 || options.DynamicRange <= 0 {
		return Options{}, ErrInvalidParameter
	}

	return *options, nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// asShallowNRGBA returns src as non-premultiplied RGBA, avoiding a copy when
// src already has that form. Working in non-premultiplied space means a
// pixel's luminance is its visible color: alpha only weights the sample and
// never dims the colors being compared.
func asShallowNRGBA(src image.Image) *image.NRGBA {
	if nrgba, ok := src.(*image.NRGBA); ok {
		return nrgba
	}
	bounds := src.Bounds()
	dst := image.NewNRGBA(bounds)
	draw.Draw(dst, bounds, src, bounds.Min, draw.Src)
	return dst
}

// isNilImage reports whether img is nil, including a nil pointer stored in a
// non-nil image.Image interface.
func isNilImage(img image.Image) bool {
	if img == nil {
		return true
	}
	value := reflect.ValueOf(img)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

// heatValue maps a local SSIM score to the 0-255 quality map.
func heatValue(score float64) uint8 {
	return uint8(math.Round(f64.Clamp(score, 0, 1) * 255))
}

const (
	momentWeight = iota
	momentRef
	momentCand
	momentRefSq
	momentCandSq
	momentRefCand
	momentCount
)

// weightEpsilon is the threshold below which a window total is treated as
// "no visible pixels". Window weights are products of two 8-bit alphas, so a
// window with any visible pixel sums to at least 1; this only absorbs
// floating point cancellation residue.
const weightEpsilon = 0.5

// windowSums maintains the weighted sums of a sliding window. It uses a
// horizontal pass per row and a vertical accumulator, so the cost of a window
// does not depend on the window size.
type windowSums struct {
	radius   int
	width    int
	acc      [momentCount][]float64
	row      [momentCount][]float64
	ref      *image.NRGBA
	cand     *image.NRGBA
	refMinX  int
	refMinY  int
	candMinX int
	candMinY int
}

func newWindowSums(width, radius int, ref, cand *image.NRGBA) *windowSums {
	s := &windowSums{
		radius:   radius,
		width:    width,
		ref:      ref,
		cand:     cand,
		refMinX:  ref.Bounds().Min.X,
		refMinY:  ref.Bounds().Min.Y,
		candMinX: cand.Bounds().Min.X,
		candMinY: cand.Bounds().Min.Y,
	}
	for i := 0; i < momentCount; i++ {
		s.acc[i] = make([]float64, width)
		s.row[i] = make([]float64, width)
	}
	return s
}

// addRow adds (sign 1) or removes (sign -1) one row from the vertical window
// accumulator.
func (s *windowSums) addRow(y int, sign float64) {
	s.computeRow(y)
	for i := 0; i < momentCount; i++ {
		acc := s.acc[i]
		row := s.row[i]
		for x := 0; x < s.width; x++ {
			acc[x] += sign * row[x]
		}
	}
}

// computeRow fills s.row with the horizontal window sums of row y, clipped at
// the left and right image edges.
func (s *windowSums) computeRow(y int) {
	width := s.width
	radius := s.radius

	refPos := s.ref.PixOffset(s.refMinX, s.refMinY+y)
	candPos := s.cand.PixOffset(s.candMinX, s.candMinY+y)

	var sum [momentCount]float64
	right := min(radius, width-1)
	for x := 0; x <= right; x++ {
		m := pixelMoments(s.ref, refPos+x*4, s.cand, candPos+x*4)
		for i := 0; i < momentCount; i++ {
			sum[i] += m[i]
		}
	}
	for i := 0; i < momentCount; i++ {
		s.row[i][0] = sum[i]
	}

	for x := 1; x < width; x++ {
		if add := x + radius; add < width {
			m := pixelMoments(s.ref, refPos+add*4, s.cand, candPos+add*4)
			for i := 0; i < momentCount; i++ {
				sum[i] += m[i]
			}
		}
		if remove := x - radius - 1; remove >= 0 {
			m := pixelMoments(s.ref, refPos+remove*4, s.cand, candPos+remove*4)
			for i := 0; i < momentCount; i++ {
				sum[i] -= m[i]
			}
		}
		for i := 0; i < momentCount; i++ {
			s.row[i][x] = sum[i]
		}
	}
}

// pixelMoments returns the weighted moments of a pair of pixels. The weight is
// the product of the two 8-bit alpha values, so it is an integer in 0..65025
// and every moment is an integer multiple of 1/100 (the luma weights have a
// denominator of 10). Sliding a window therefore adds and removes the same
// quantities, and a window containing only invisible pixels cancels exactly.
func pixelMoments(ref *image.NRGBA, refPos int, cand *image.NRGBA, candPos int) [momentCount]float64 {
	var m [momentCount]float64

	weight := float64(int(ref.Pix[refPos+3]) * int(cand.Pix[candPos+3]))
	if weight == 0 {
		return m
	}

	refLuma := lumaR*float64(ref.Pix[refPos+0]) +
		lumaG*float64(ref.Pix[refPos+1]) +
		lumaB*float64(ref.Pix[refPos+2])
	candLuma := lumaR*float64(cand.Pix[candPos+0]) +
		lumaG*float64(cand.Pix[candPos+1]) +
		lumaB*float64(cand.Pix[candPos+2])

	m[momentWeight] = weight
	m[momentRef] = weight * refLuma
	m[momentCand] = weight * candLuma
	m[momentRefSq] = weight * refLuma * refLuma
	m[momentCandSq] = weight * candLuma * candLuma
	m[momentRefCand] = weight * refLuma * candLuma

	return m
}
