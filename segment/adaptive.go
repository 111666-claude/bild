package segment

import (
	"errors"
	"image"
	"image/color"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
	"github.com/anthonynsimon/bild/util"
)

// Recommended defaults for Sauvola adaptive thresholding. The dynamic range
// matches the 0-255 luminance scale produced by util.Rank.
const (
	// DefaultSauvolaWindow is the default side length, in pixels, of the
	// square neighborhood used to estimate the local mean and variance.
	DefaultSauvolaWindow = 25
	// DefaultSauvolaK is the default Sauvola sensitivity factor.
	DefaultSauvolaK = 0.2
	// DefaultSauvolaR is the default dynamic range of the standard deviation.
	DefaultSauvolaR = 128.0
)

var (
	errOddWindow      = errors.New("segment: window size must be a positive odd integer")
	errNonFiniteK     = errors.New("segment: k must be a finite number")
	errNonPositiveR   = errors.New("segment: dynamic range r must be a finite positive number")
	errWindowTooLarge = errors.New("segment: window size must not be larger than the image")
)

// SauvolaThreshold applies adaptive (Sauvola) binarization, which is well
// suited to line art photographed or scanned under uneven illumination: a
// single global level loses strokes on either the shadowed or overexposed
// side, while a per-pixel threshold tracks the local background.
//
// For every pixel the threshold is computed from the luminance mean m and
// standard deviation s of its (2*radius+1) squared neighborhood:
//
//	t = m * (1 + k * (s/r - 1))
//
// Pixels whose luminance is below t become black (foreground) and the rest
// become white (background). Fully transparent pixels are always white and
// are excluded from the neighborhood statistics, matching Threshold's
// handling of transparency.
//
// window is the neighborhood side length and must be a positive odd integer.
// k is the Sauvola sensitivity factor (typical values lie around 0.2) and
// must be finite. r is the maximum expected standard deviation of the
// luminance values (128 for an 8-bit grayscale image) and must be finite and
// positive. Neighborhoods that extend past the image edge are clipped to the
// intersection with the image bounds, so border pixels use a smaller but
// fully in-bounds window.
//
// The returned image is a *image.Gray whose Rect equals the input bounds.
// An error is returned for an even or non-positive window, non-finite k, a
// non-finite or non-positive r, or a window larger than either dimension of
// the input image; the returned image is nil in that case. An empty input
// image yields an empty *image.Gray and no error. The input image is never
// modified.
//
// Using summed-area tables, running time is O(W*H) regardless of the window
// size, and the extra memory is O(W*H): three integral images of float64
// values plus the output Gray image, all linear in the image area.
func SauvolaThreshold(img image.Image, window int, k, r float64) (*image.Gray, error) {
	if window <= 0 || window%2 == 0 {
		return nil, errOddWindow
	}
	if math.IsNaN(k) || math.IsInf(k, 0) {
		return nil, errNonFiniteK
	}
	if math.IsNaN(r) || math.IsInf(r, 0) || r <= 0 {
		return nil, errNonPositiveR
	}

	src := clone.AsRGBA(img)
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w == 0 || h == 0 {
		return image.NewGray(bounds), nil
	}
	if window > w || window > h {
		return nil, errWindowTooLarge
	}

	radius := window / 2

	// Summed-area tables over the non-transparent pixels of the source.
	// Integral cell (i, j) accumulates every pixel in rows [0, i) and
	// columns [0, j), so a rectangular query is a constant-time combination
	// of four table lookups. count records how many opaque pixels were
	// accumulated, sum their luminance and sqSum the squared luminance.
	// clone.AsRGBA normalizes the copy to a zero origin, so table and source
	// indexing are relative to (0,0); the output Gray image is created with
	// the original bounds, so a non-zero origin passes through untouched.
	tableW, tableH := w+1, h+1
	count := make([]float64, tableW*tableH)
	sum := make([]float64, tableW*tableH)
	sqSum := make([]float64, tableW*tableH)

	for row := 0; row < h; row++ {
		var rowCount, rowSum, rowSqSum float64
		srcRowStart := row * src.Stride
		tableRowStart := (row + 1) * tableW
		prevRowStart := row * tableW
		for col := 0; col < w; col++ {
			pos := srcRowStart + col*4
			c := src.Pix[pos : pos+4]
			if !(c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0) {
				lum := util.Rank(color.RGBA{c[0], c[1], c[2], c[3]})
				rowCount++
				rowSum += lum
				rowSqSum += lum * lum
			}
			cell := tableRowStart + col + 1
			count[cell] = count[prevRowStart+col+1] + rowCount
			sum[cell] = sum[prevRowStart+col+1] + rowSum
			sqSum[cell] = sqSum[prevRowStart+col+1] + rowSqSum
		}
	}

	dst := image.NewGray(bounds)

	// Classification is independent per pixel, so rows can be split across
	// CPUs. Every goroutine only reads shared tables and writes its own
	// output rows, which keeps the result deterministic.
	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			top := y - radius
			if top < 0 {
				top = 0
			}
			bottom := y + radius + 1
			if bottom > h {
				bottom = h
			}
			topRow := top * tableW
			bottomRow := bottom * tableW

			for x := 0; x < w; x++ {
				dstPos := y*dst.Stride + x
				srcPos := y*src.Stride + x*4
				c := src.Pix[srcPos : srcPos+4]

				// A fully transparent pixel is always background.
				if c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0 {
					dst.Pix[dstPos] = 0xFF
					continue
				}

				left := x - radius
				if left < 0 {
					left = 0
				}
				right := x + radius + 1
				if right > w {
					right = w
				}

				n := count[bottomRow+right] - count[topRow+right] -
					count[bottomRow+left] + count[topRow+left]
				s := sum[bottomRow+right] - sum[topRow+right] -
					sum[bottomRow+left] + sum[topRow+left]
				sq := sqSum[bottomRow+right] - sqSum[topRow+right] -
					sqSum[bottomRow+left] + sqSum[topRow+left]

				// A window reaching this point always contains at least the
				// current opaque pixel, hence n >= 1.
				mean := s / n
				variance := sq/n - mean*mean
				if variance < 0 {
					// Rounding error in the squared sums can only drive the
					// variance slightly below zero.
					variance = 0
				}
				stdDev := math.Sqrt(variance)
				threshold := mean * (1 + k*(stdDev/r-1))

				lum := util.Rank(color.RGBA{c[0], c[1], c[2], c[3]})
				if lum < threshold {
					dst.Pix[dstPos] = 0x00
				} else {
					dst.Pix[dstPos] = 0xFF
				}
			}
		}
	})

	return dst, nil
}
