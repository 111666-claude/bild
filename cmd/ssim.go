package cmd

import (
	"fmt"
	"image"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/ssim"
	"github.com/spf13/cobra"
)

// ssimFlags holds the comparison parameters shared by the ssim subcommands.
type ssimFlags struct {
	window       int
	k1           float64
	k2           float64
	dynamicRange float64
}

func (f *ssimFlags) register(cmd *cobra.Command) {
	cmd.Flags().IntVarP(&f.window, "window", "w", ssim.DefaultWindowSize, "side length of the square comparison window, must be odd")
	cmd.Flags().Float64Var(&f.k1, "k1", ssim.DefaultK1, "stabilizer constant for the luminance term")
	cmd.Flags().Float64Var(&f.k2, "k2", ssim.DefaultK2, "stabilizer constant for the contrast and structure terms")
	cmd.Flags().Float64Var(&f.dynamicRange, "dynamic-range", ssim.DefaultDynamicRange, "maximum luma value")
}

func (f *ssimFlags) options() *ssim.Options {
	return &ssim.Options{
		WindowSize:   f.window,
		K1:           f.k1,
		K2:           f.k2,
		DynamicRange: f.dynamicRange,
	}
}

func openSSIMInputs(referencePath, candidatePath string) (image.Image, image.Image) {
	reference, err := imgio.Open(referencePath)
	exitIfNotNil(err)
	candidate, err := imgio.Open(candidatePath)
	exitIfNotNil(err)
	return reference, candidate
}

func ssimCompare() *cobra.Command {
	var (
		flags       ssimFlags
		heatmapPath string
		colorize    bool
	)

	var cmd = &cobra.Command{
		Use:     "compare",
		Short:   "score a candidate image against a reference using local SSIM",
		Args:    cobra.ExactArgs(2),
		Example: "compare --heatmap heat.png reference.png candidate.png",
		Run: func(cmd *cobra.Command, args []string) {
			reference, candidate := openSSIMInputs(args[0], args[1])

			result, err := ssim.Compare(reference, candidate, flags.options())
			exitIfNotNil(err)

			// The scores go to stdout so they can be captured by scripts.
			fmt.Printf("score=%.6f\n", result.Score)
			fmt.Printf("luminance=%.6f\n", result.Luminance)
			fmt.Printf("contrast=%.6f\n", result.Contrast)
			fmt.Printf("structure=%.6f\n", result.Structure)

			if heatmapPath != "" {
				saveHeatmap(heatmapPath, result.Heatmap, colorize)
			}
		}}

	flags.register(cmd)
	cmd.Flags().StringVar(&heatmapPath, "heatmap", "", "write a per-pixel quality heatmap to this file")
	cmd.Flags().BoolVar(&colorize, "colorize", false, "write the heatmap with a sequential color ramp instead of grayscale")

	return cmd
}

func ssimHeatmap() *cobra.Command {
	var (
		flags    ssimFlags
		colorize bool
	)

	var cmd = &cobra.Command{
		Use:     "heatmap",
		Short:   "write a local SSIM quality heatmap for a candidate image",
		Args:    cobra.ExactArgs(3),
		Example: "heatmap reference.png candidate.png heatmap.png",
		Run: func(cmd *cobra.Command, args []string) {
			reference, candidate := openSSIMInputs(args[0], args[1])

			result, err := ssim.Compare(reference, candidate, flags.options())
			exitIfNotNil(err)

			saveHeatmap(args[2], result.Heatmap, colorize)
		}}

	flags.register(cmd)
	cmd.Flags().BoolVar(&colorize, "colorize", false, "write the heatmap with a sequential color ramp instead of grayscale")

	return cmd
}

// saveHeatmap writes the quality map, keeping the output file extension based
// encoder selection used by the rest of the CLI.
func saveHeatmap(filename string, heatmap *image.Gray, colorize bool) {
	encoder := resolveEncoder(filename, imgio.PNGEncoder())
	if colorize {
		exitIfNotNil(imgio.Save(filename, ssim.ColorizeHeatmap(heatmap), encoder))
		return
	}
	exitIfNotNil(imgio.Save(filename, heatmap, encoder))
}

func createSSIM() *cobra.Command {
	var cmd = &cobra.Command{
		Use:   "ssim",
		Short: "local structural similarity comparison of images",
	}

	cmd.AddCommand(ssimCompare())
	cmd.AddCommand(ssimHeatmap())

	return cmd
}
