package cmd

import (
	"fmt"
	"image"
	"image/color"
	"strconv"
	"strings"

	"github.com/anthonynsimon/bild/halftone"
	"github.com/spf13/cobra"
)

// parseHexColor parses a single CSS-like hex color ("#rgb", "rgb", "#rgba",
// "#rrggbb" or "#rrggbbaa") into a non-premultiplied color.RGBA.
func parseHexColor(input string) (color.RGBA, error) {
	hex := strings.TrimPrefix(input, "#")
	compact := len(hex) == 3 || len(hex) == 4
	full := len(hex) == 6 || len(hex) == 8
	if !compact && !full {
		return color.RGBA{}, fmt.Errorf("invalid color %q: use #rgb, #rgba, #rrggbb or #rrggbbaa", input)
	}

	expand := func(c byte) string {
		return strings.Repeat(string(c), 2)
	}

	var rr, gg, bb, aa string
	if compact {
		rr, gg, bb = expand(hex[0]), expand(hex[1]), expand(hex[2])
		if len(hex) == 4 {
			aa = expand(hex[3])
		} else {
			aa = "ff"
		}
	} else {
		rr, gg, bb = hex[0:2], hex[2:4], hex[4:6]
		if len(hex) == 8 {
			aa = hex[6:8]
		} else {
			aa = "ff"
		}
	}

	values := [4]string{rr, gg, bb, aa}
	var out [4]uint8
	for i, v := range values {
		n, err := strconv.ParseUint(v, 16, 8)
		if err != nil {
			return color.RGBA{}, fmt.Errorf("invalid color %q: %w", input, err)
		}
		out[i] = uint8(n)
	}

	return color.RGBA{R: out[0], G: out[1], B: out[2], A: out[3]}, nil
}

// parsePalette parses a comma separated list of hex colors into a palette.
func parsePalette(input string) (color.Palette, error) {
	palette := color.Palette{}
	for _, token := range strings.Split(input, ",") {
		token = strings.TrimSpace(token)
		if token == "" {
			continue
		}
		c, err := parseHexColor(token)
		if err != nil {
			return nil, err
		}
		palette = append(palette, c)
	}
	if len(palette) == 0 {
		return nil, fmt.Errorf("palette must contain at least one color")
	}
	return palette, nil
}

func ordered() *cobra.Command {
	var (
		matrixSize int
		strength   float64
		originStr  string
		paletteStr string
	)

	var cmd = &cobra.Command{
		Use:     "ordered",
		Short:   "apply ordered halftone (ordered dithering), restricting the image to black and white or a custom palette",
		Args:    cobra.ExactArgs(2),
		Example: "ordered --matrix-size 8 --strength 1 input.jpg output.png",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fout := args[1]

			apply(fin, fout, func(img image.Image) (image.Image, error) {
				origin := image.Point{}
				if originStr != "" {
					size, err := parseSizeStr(originStr)
					if err != nil {
						return nil, err
					}
					origin = image.Pt(size.Width, size.Height)
				}

				opts := &halftone.Options{
					MatrixSize: matrixSize,
					Strength:   strength,
					Origin:     origin,
				}

				if paletteStr != "" {
					palette, err := parsePalette(paletteStr)
					if err != nil {
						return nil, err
					}
					opts.Palette = palette
				}

				return halftone.Ordered(img, opts)
			})
		}}

	cmd.Flags().IntVarP(&matrixSize, "matrix-size", "m", 8, "side of the square Bayer threshold matrix (1 to 256)")
	cmd.Flags().Float64VarP(&strength, "strength", "s", 1, "dithering strength, from a small value (plain quantization) to 1 (full dither)")
	cmd.Flags().StringVarP(&originStr, "origin", "o", "", "matrix phase origin as XxY, i.e. 2x2; defaults to the image origin")
	cmd.Flags().StringVarP(&paletteStr, "palette", "p", "", "comma separated hex colors to restrict the output to, i.e. #000000,#ffffff")

	return cmd
}

func createHalftone() *cobra.Command {
	var cmd = &cobra.Command{
		Use:   "halftone",
		Short: "apply halftone dithering on images",
	}

	cmd.AddCommand(ordered())

	return cmd
}
