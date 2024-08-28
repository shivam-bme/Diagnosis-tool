# convert_mha_to_png.py
import SimpleITK as sitk
import sys

def convert_mha_to_png(mha_path, output_path):
    # Read the MHA file
    image = sitk.ReadImage(mha_path)
    
    # Convert to PNG
    sitk.WriteImage(image, output_path)
    return sitk.WriteImage(image, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_mha_to_png.py <input_mha> <output_png>")
        sys.exit(1)

    mha_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_mha_to_png(mha_path, output_path)
