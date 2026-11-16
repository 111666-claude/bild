package cmd

import (
	"image"

	"github.com/anthonynsimon/bild/segment"
	"github.com/spf13/cobra"
)

func threshold() *cobra.Command {
	var level uint8

	var cmd = &cobra.Command{
		Use:     "threshold",
		Short:   "segment an image by a threshold",
		Args:    cobra.ExactArgs(2),
		Example: "threshold --level 200 input.jpg output.jpg",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fout := args[1]

			apply(fin, fout, func(img image.Image) (image.Image, error) {
				return segment.Threshold(img, level), nil
			})
		}}

	cmd.Flags().Uint8VarP(&level, "level", "l", 128, "the level at which the segmenting threshold will be crossed")

	return cmd
}

func sauvola() *cobra.Command {
	var window int
	var k, r float64

	var cmd = &cobra.Command{
		Use:     "sauvola",
		Short:   "segment an image with adaptive Sauvola thresholding",
		Args:    cobra.ExactArgs(2),
		Example: "sauvola --window 25 --k 0.2 --r 128 input.jpg output.jpg",
		Run: func(cmd *cobra.Command, args []string) {
			fin := args[0]
			fout := args[1]

			apply(fin, fout, func(img image.Image) (image.Image, error) {
				return segment.SauvolaThreshold(img, window, k, r)
			})
		}}

	cmd.Flags().IntVarP(&window, "window", "w", segment.DefaultSauvolaWindow, "odd side length of the local neighborhood window")
	cmd.Flags().Float64VarP(&k, "k", "k", segment.DefaultSauvolaK, "Sauvola sensitivity factor")
	cmd.Flags().Float64VarP(&r, "r", "r", segment.DefaultSauvolaR, "dynamic range of the local standard deviation")

	return cmd
}

func createSegment() *cobra.Command {
	var blurCmd = &cobra.Command{
		Use:   "segment",
		Short: "segment an image using the specified method",
	}

	blurCmd.AddCommand(threshold())
	blurCmd.AddCommand(sauvola())

	return blurCmd
}
