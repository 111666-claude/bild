package cmd

import (
	"image"

	"github.com/anthonynsimon/bild/blur"
	"github.com/spf13/cobra"
)

func box() *cobra.Command {
	var radius float64

	var cmd = &cobra.Command{
		Use:     "box",
		Short:   "apply box blur to an input image",
		Args:    cobra.ExactArgs(2),
		Example: "box --radius 0.5 input.jpg output.jpg",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fout := args[1]

			apply(fin, fout, func(img image.Image) (image.Image, error) {
				return blur.Box(img, radius), nil
			})
		}}

	cmd.Flags().Float64VarP(&radius, "radius", "r", 0, "the blur's radius")

	return cmd
}

func gaussian() *cobra.Command {
	var radius float64

	var cmd = &cobra.Command{
		Use:     "gaussian",
		Short:   "apply gaussian blur to an input image",
		Args:    cobra.ExactArgs(2),
		Example: "gaussian --radius 0.5 input.jpg output.jpg",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fout := args[1]

			apply(fin, fout, func(img image.Image) (image.Image, error) {
				return blur.Gaussian(img, radius), nil
			})
		}}

	cmd.Flags().Float64VarP(&radius, "radius", "r", 0, "the blur's radius")

	return cmd
}

func bilateral() *cobra.Command {
	var radius, spatialSigma, colorSigma float64

	var cmd = &cobra.Command{
		Use:     "bilateral",
		Short:   "apply an edge-preserving bilateral blur to an input image",
		Args:    cobra.ExactArgs(2),
		Example: "bilateral --radius 3 --spatial-sigma 3 --color-sigma 25 input.jpg output.jpg",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fout := args[1]

			apply(fin, fout, func(img image.Image) (image.Image, error) {
				return blur.Bilateral(img, radius, spatialSigma, colorSigma)
			})
		}}

	cmd.Flags().Float64VarP(&radius, "radius", "r", 3, "the size of the neighborhood in pixels")
	cmd.Flags().Float64VarP(&spatialSigma, "spatial-sigma", "s", 3, "the spatial weight falloff")
	cmd.Flags().Float64VarP(&colorSigma, "color-sigma", "c", 25, "the color weight falloff, in the 0 to 255 range")

	return cmd
}

func createBlur() *cobra.Command {
	var blurCmd = &cobra.Command{
		Use:   "blur",
		Short: "blur an image using the specified method",
	}

	blurCmd.AddCommand(box())
	blurCmd.AddCommand(gaussian())
	blurCmd.AddCommand(bilateral())

	return blurCmd
}
