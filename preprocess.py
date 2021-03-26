import argparse
from glob import glob
from html2df import HTML2df

def main(parser):
    args = parser.parse_args()
    html_processer = HTML2df()
    for filepath in glob(args.input_path + "*." + args.file_type):
        filename = filepath.split("/")[-1]
        df = html_processer.convert2df(filepath, generate_label=args.with_label)
        df.to_csv(args.output_path + filename.split(".")[0] + ".csv", index=False)
        print(filename, "processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Set up input folder.", required=True)
    parser.add_argument("--output_path", type=str, help="Set up output folder.", required=True)
    parser.add_argument("--file_type", type=str, help="Set input file type (html or txt).", default="txt")
    parser.add_argument("--with_label", action="store_true", help="Set to enable label generation", default=False)
    main(parser)