/*
Package halftone provides ordered dithering (ordered halftone) methods.

Ordered halftone reduces an image to a restricted palette — by default pure
black and white, as needed by e-ink displays — while preserving apparent
shading through a threshold matrix that repeats contiguously across the whole
image. Because the matrix is indexed by the image's absolute coordinates,
processing a cropped sub-image produces exactly the pixels the full-image
processing would have produced for that region.
*/
package halftone

import (
	"errors"
	"image"
	"image/color"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
	"github.com/anthonynsimon/bild/util"
)

// MaxMatrixSize is the largest accepted threshold matrix side. A matrix holds
// size*size float64 values, so 256 keeps its memory bounded (~0.5 MiB).
const MaxMatrixSize = 256

var (
	// ErrNilImage is returned when a nil image is provided.
	ErrNilImage = errors.New("halftone: input image is nil")
	// ErrInvalidMatrixSize is returned when the matrix side is negative or
	// larger than MaxMatrixSize. A zero value selects the default size.
	ErrInvalidMatrixSize = errors.New("halftone: matrix size must be between 1 and MaxMatrixSize")
	// ErrInvalidStrength is returned when the strength is not a finite number.
	ErrInvalidStrength = errors.New("halftone: strength must be a finite number")
	// ErrInvalidPalette is returned when the palette is empty.
	ErrInvalidPalette = errors.New("halftone: palette must contain at least one color")
)

// Options configures the ordered halftone operation.
//
// MatrixSize is the side of the square Bayer threshold matrix generated and
// tiled over the image. It must be between 1 and MaxMatrixSize; a zero value
// defaults to 8. Larger matrices render more gray levels at the cost of a
// coarser dot pattern.
//
// Strength scales how strongly the threshold matrix perturbs each pixel and
// must be finite. A value of 1.0 is the standard ordered dither; the zero
// value is treated as unset and also selects full strength, so a zero-valued
// Options struct gets the documented default. Pass an explicit small value to
// weaken the dither, approaching plain quantization to the closest palette
// entry as strength nears zero. Negative values behave like zero and values
// above one behave like one.
//
// Origin shifts the matrix phase. The matrix cell used for a pixel is taken
// from the pixel's absolute image coordinates relative to Origin, so cropping
// and tiling remain consistent with full-image processing.
//
// Palette holds the colors the output is restricted to. When nil, a black and
// white palette is used. Fully transparent input pixels (alpha == 0) are
// always preserved as transparent, independent of the palette.
type Options struct {
	MatrixSize int
	Strength   float64
	Origin     image.Point
	Palette    color.Palette
}

// paletteEntry pairs a palette color with its perceived luminance so that the
// per-pixel nearest-color lookup does not recompute it.
type paletteEntry struct {
	rgba color.RGBA
	lum  float64
}

// Ordered applies ordered halftone (ordered dithering) to img and returns the
// result as a new *image.RGBA restricted to the configured palette.
//
// With a nil options pointer, an 8x8 Bayer matrix, full strength and a black
// and white palette are used. An error is returned for a nil image or invalid
// options; an image with empty bounds yields an empty RGBA image.
func Ordered(img image.Image, opts *Options) (*image.RGBA, error) {
	if img == nil {
		return nil, ErrNilImage
	}

	bounds := img.Bounds()
	if bounds.Empty() {
		return image.NewRGBA(bounds), nil
	}

	if opts == nil {
		opts = &Options{}
	}

	size := opts.MatrixSize
	if size == 0 {
		size = 8
	}
	if size < 0 || size > MaxMatrixSize {
		return nil, ErrInvalidMatrixSize
	}
	if math.IsNaN(opts.Strength) || math.IsInf(opts.Strength, 0) {
		return nil, ErrInvalidStrength
	}

	palette := opts.Palette
	if palette == nil {
		palette = color.Palette{color.Black, color.White}
	}
	if len(palette) == 0 {
		return nil, ErrInvalidPalette
	}

	// A zero strength is the unset (Go zero) value and selects full strength,
	// matching the documented default. Pass an explicit small value to weaken
	// the dither down to plain palette quantization.
	strength := opts.Strength
	if strength == 0 {
		strength = 1
	}
	if strength < 0 {
		strength = 0
	} else if strength > 1 {
		strength = 1
	}

	matrix, err := BayerMatrix(size)
	if err != nil {
		return nil, err
	}

	// Precompute the luminance of every palette entry and keep the palette
	// order as the deterministic tie breaker.
	entries := make([]paletteEntry, len(palette))
	for i, c := range palette {
		r, g, b, a := c.RGBA()
		rgba := color.RGBA{
			R: uint8(r >> 8),
			G: uint8(g >> 8),
			B: uint8(b >> 8),
			A: uint8(a >> 8),
		}
		entries[i] = paletteEntry{
			rgba: rgba,
			lum:  util.Rank(rgba),
		}
	}

	src := clone.AsRGBA(img)
	dst := image.NewRGBA(bounds)
	w, h := bounds.Dx(), bounds.Dy()
	ox, oy := opts.Origin.X, opts.Origin.Y

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				pos := y*src.Stride + x*4
				alpha := src.Pix[pos+3]

				dstPos := y*dst.Stride + x*4

				// Fully transparent pixels are preserved as-is.
				if alpha == 0 {
					dst.Pix[dstPos+0] = 0
					dst.Pix[dstPos+1] = 0
					dst.Pix[dstPos+2] = 0
					dst.Pix[dstPos+3] = 0
					continue
				}

				pixel := color.RGBA{
					R: src.Pix[pos+0],
					G: src.Pix[pos+1],
					B: src.Pix[pos+2],
					A: alpha,
				}
				lum := util.Rank(pixel)

				// Index the tiled matrix by absolute coordinates so that
				// cropped blocks match full-image processing.
				mx := mod(bounds.Min.X+x-ox, size)
				my := mod(bounds.Min.Y+y-oy, size)
				threshold := matrix[my*size+mx]

				// The matrix threshold is centered around mid-gray. With
				// strength 1 it spans the full [0, 255] range; with strength
				// 0 the perturbation vanishes and plain quantization remains.
				adjusted := lum + (threshold-127.5)*strength
				if adjusted < 0 {
					adjusted = 0
				} else if adjusted > 255 {
					adjusted = 255
				}

				best := nearestPaletteEntry(entries, adjusted)
				dst.Pix[dstPos+0] = best.R
				dst.Pix[dstPos+1] = best.G
				dst.Pix[dstPos+2] = best.B
				dst.Pix[dstPos+3] = best.A
			}
		}
	})

	return dst, nil
}

// nearestPaletteEntry returns the palette entry whose luminance is closest to
// target. Palette order breaks ties, keeping the result deterministic.
func nearestPaletteEntry(entries []paletteEntry, target float64) color.RGBA {
	best := entries[0].rgba
	bestDist := math.Abs(entries[0].lum - target)
	for i := 1; i < len(entries); i++ {
		dist := math.Abs(entries[i].lum - target)
		if dist < bestDist {
			best = entries[i].rgba
			bestDist = dist
		}
	}
	return best
}

// BayerMatrix builds a square ordered-dithering threshold matrix of the given
// side. The returned values are centered around mid-gray and strictly within
// the (0, 255) range, holding size*size evenly spaced threshold levels, each
// used exactly once. Centering the levels guarantees that solid black and
// solid white inputs remain solid regardless of the matrix cell they hit.
//
// Sizes that are powers of two produce the classic Bayer matrix through
// recursive tiling. Any other positive size up to MaxMatrixSize is supported
// by ranking the bit-interleaved coordinates; ties between colliding ranks
// are broken deterministically so that every cell still gets a distinct level.
func BayerMatrix(size int) ([]float64, error) {
	if size <= 0 || size > MaxMatrixSize {
		return nil, ErrInvalidMatrixSize
	}

	ranks := make([]int, size*size)
	if size&(size-1) == 0 {
		// Classic recursive Bayer matrix. Start from the single-cell matrix
		// and double the side on every step, expanding each previous cell v
		// into the four quadrants with distinct offsets:
		//
		//	[ 4*v+0  4*v+2 ]
		//	[ 4*v+3  4*v+1 ]
		//
		ranks[0] = 0
		for n := 1; n < size; n *= 2 {
			next := make([]int, 4*n*n)
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					v := ranks[y*n+x] * 4
					next[(2*y)*2*n+(2*x)] = v + 0
					next[(2*y)*2*n+(2*x+1)] = v + 2
					next[(2*y+1)*2*n+(2*x)] = v + 3
					next[(2*y+1)*2*n+(2*x+1)] = v + 1
				}
			}
			ranks = next
		}
	} else {
		// Generic fallback: rank cells by bit-interleaved coordinates, then
		// resolve collisions by assigning the next free rank. The rank count
		// equals the cell count, so a free rank always exists.
		used := make(map[int]bool, size*size)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				rank := interleave(x, y)
				for used[rank] {
					rank++
				}
				used[rank] = true
				ranks[y*size+x] = rank
			}
		}
		// Compact arbitrary ranks into the contiguous [0, size*size) range.
		order := make([]int, len(ranks))
		for i := range order {
			order[i] = i
		}
		for i := 1; i < len(order); i++ {
			for j := i; j > 0 && ranks[order[j-1]] > ranks[order[j]]; j-- {
				order[j-1], order[j] = order[j], order[j-1]
			}
		}
		for level, idx := range order {
			ranks[idx] = level
		}
	}

	// Scale ranks to evenly spaced thresholds centered on mid-gray. Adding
	// half a cell keeps every threshold strictly inside (0, 255), so the
	// extreme black and white tones are never flipped by a boundary cell.
	cells := float64(size * size)
	matrix := make([]float64, size*size)
	scale := 256.0 / cells
	for i, rank := range ranks {
		matrix[i] = (float64(rank) + 0.5) * scale
	}
	return matrix, nil
}

// interleave returns the rank of the point (x, y) by interleaving the bits of
// the two coordinates. For power-of-two sizes this is the Bayer ordering.
func interleave(x, y int) int {
	rank := 0
	for bit := 0; bit < 16; bit++ {
		rank |= ((x >> bit) & 1) << (2 * bit)
		rank |= ((y >> bit) & 1) << (2*bit + 1)
	}
	return rank
}

// mod returns the Euclidean remainder of a over b, always non-negative. This
// lets negative coordinates (for example from a non-zero Bounds.Min or an
// Origin shift) index the tiled matrix correctly.
func mod(a, n int) int {
	r := a % n
	if r < 0 {
		r += n
	}
	return r
}
