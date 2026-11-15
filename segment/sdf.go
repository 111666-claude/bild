package segment

import (
	"image"
	"math"

	"github.com/anthonynsimon/bild/clone"
	"github.com/anthonynsimon/bild/parallel"
)

// SDF returns a single-channel signed distance field generated from the alpha
// channel of img, suitable for rendering shapes and icons that stay crisp when
// scaled.
//
// Pixels with alpha >= threshold are considered inside the shape. The result
// has the same bounds as img and encodes the signed distance to the nearest
// shape boundary: the contour is encoded as 128, distances inside the shape
// map to values greater than 128 and distances outside map to values smaller
// than 128, varying monotonically with the distance. The spread parameter is
// the distance in pixels over which the field ramps to its extremes: pixels at
// distance >= spread from the boundary saturate to 255 (inside) or 0
// (outside). If spread <= 0 it is treated as 1.
//
// Pixels outside the image bounds are treated as if the edge pixels extended
// outwards, so a shape touching the image border does not produce a spurious
// contour there. Semi-transparent pixels (0 < alpha < 255) lying on the
// boundary shift the field by a sub-pixel amount derived from their coverage.
//
// If no pixel is inside the shape the result is uniformly 0; if every pixel
// is inside the result is uniformly 255.
//
// The distance field is computed with a linear-time exact Euclidean distance
// transform, so the cost and memory grow proportionally to the pixel count.
func SDF(img image.Image, threshold uint8, spread float64) *image.Gray {
	if spread <= 0 {
		spread = 1
	}

	src := clone.AsRGBA(img)
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	inside := make([]bool, w*h)
	insideCount := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if src.Pix[y*src.Stride+x*4+3] >= threshold {
				inside[y*w+x] = true
				insideCount++
			}
		}
	}

	dst := image.NewGray(bounds)

	if insideCount == 0 {
		// No shape: everything is as far outside as the field reaches.
		return dst
	}
	if insideCount == w*h {
		// No background: everything is as far inside as the field reaches.
		for i := range dst.Pix {
			dst.Pix[i] = 0xFF
		}
		return dst
	}

	// Squared distances to the nearest inside and outside pixel.
	dIn := edt(inside, w, h, true)
	dOut := edt(inside, w, h, false)

	parallel.Line(h, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				i := y*w + x
				signed := math.Sqrt(dOut[i]) - math.Sqrt(dIn[i])

				// Sub-pixel refinement for semi-transparent boundary pixels.
				// Note the denominators can't be zero: threshold == 255 forces
				// the inside branch to be unreachable for alpha < 255, and
				// threshold == 0 makes every pixel inside.
				a := src.Pix[y*src.Stride+x*4+3]
				if a > 0 && a < 0xFF {
					if a >= threshold {
						if signed <= math.Sqrt2 {
							signed = float64(a-threshold) / float64(0xFF-threshold)
						}
					} else if signed >= -math.Sqrt2 {
						signed = -float64(threshold-a) / float64(threshold)
					}
				}

				norm := signed / spread
				norm = math.Max(-1, math.Min(1, norm))
				dst.Pix[y*dst.Stride+x] = uint8(math.Round(127.5 + 127.5*norm))
			}
		}
	})

	return dst
}

// edt returns the squared Euclidean distance transform of a binary grid of
// width w and height h using the linear-time algorithm of Felzenszwalb and
// Huttenlocher. If foreground is true the result holds, per cell, the squared
// distance to the nearest true cell, otherwise to the nearest false cell.
// The grid must contain at least one cell of the queried class.
func edt(grid []bool, w, h int, foreground bool) []float64 {
	// Large finite stand-in for infinity: big enough to lose against any real
	// distance, small enough to keep the parabola arithmetic free of NaNs.
	const inf = 1e18

	f := make([]float64, w*h)
	parallel.Line(w*h, func(start, end int) {
		for i := start; i < end; i++ {
			if grid[i] == foreground {
				f[i] = 0
			} else {
				f[i] = inf
			}
		}
	})

	// First pass: 1D transform along each column.
	parallel.Line(w, func(start, end int) {
		col := make([]float64, h)
		d := make([]float64, h)
		v := make([]int, h)
		z := make([]float64, h+1)
		for x := start; x < end; x++ {
			for y := 0; y < h; y++ {
				col[y] = f[y*w+x]
			}
			dt1d(col, d, v, z)
			for y := 0; y < h; y++ {
				f[y*w+x] = d[y]
			}
		}
	})

	// Second pass: 1D transform along each row.
	parallel.Line(h, func(start, end int) {
		d := make([]float64, w)
		v := make([]int, w)
		z := make([]float64, w+1)
		for y := start; y < end; y++ {
			row := f[y*w : y*w+w]
			dt1d(row, d, v, z)
			copy(row, d)
		}
	})

	return f
}

// dt1d computes the 1D squared distance transform of f into d. v and z are
// scratch buffers of length len(f) and len(f)+1 respectively.
func dt1d(f, d []float64, v []int, z []float64) {
	n := len(f)
	k := 0
	v[0] = 0
	z[0] = math.Inf(-1)
	z[1] = math.Inf(1)

	for q := 1; q < n; q++ {
		s := (f[q] - f[v[k]] + float64((q-v[k])*(q+v[k]))) / float64(2*(q-v[k]))
		for s <= z[k] {
			k--
			s = (f[q] - f[v[k]] + float64((q-v[k])*(q+v[k]))) / float64(2*(q-v[k]))
		}
		k++
		v[k] = q
		z[k] = s
		z[k+1] = math.Inf(1)
	}

	k = 0
	for q := 0; q < n; q++ {
		for z[k+1] < float64(q) {
			k++
		}
		dx := q - v[k]
		d[q] = float64(dx*dx) + f[v[k]]
	}
}
