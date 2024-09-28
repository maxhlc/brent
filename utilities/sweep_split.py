# Standard imports
import argparse
import copy
import json


def main(input: str) -> None:
    # Load sweep configuration
    with open(input) as fp:
        config = json.load(fp)

    # Split by object
    configs = []
    for ispacecraft in config["spacecraft"]:
        config_ = copy.deepcopy(config)
        config_["spacecraft"] = [ispacecraft]
        configs.append(config_)

    # Save separated configurations
    for idx, iconfig in enumerate(configs):
        # Generate filename
        output_path = input + f"_split_{idx:03d}.json"

        # Save configuration
        with open(output_path, "w") as fp:
            json.dump(iconfig, fp, indent=4)


if __name__ == "__main__":
    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser_args = parser.parse_args()

    # Execute main function
    main(parser_args.input)
