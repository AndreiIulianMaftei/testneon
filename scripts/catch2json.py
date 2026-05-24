# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import xmltodict
import json
import os


def parse_xml_dict(d):
    """takes the catch2 xml dict, performs clean-up and returns a
    list of records"""
    data = d["Catch2TestRun"]["TestCase"]
    if isinstance(data, dict):
        data = [data]
    records = []
    for cases in data:
        test_case_raw_name = cases["@name"].split(" - ")
        test_case = test_case_raw_name[0]
        value_type = test_case_raw_name[-1] if len(test_case_raw_name) > 1 else ""
        name_parts = test_case.split("::")
        benchmark_name = name_parts[0]
        dimension = name_parts[1] if len(name_parts) > 1 else ""

        sections = cases["Section"]
        if isinstance(sections, dict):
            sections = [sections]
        for section in sections:
            section_parts = section["@name"].split(" - ", 1)
            size = section_parts[0]
            variant = section_parts[1] if len(section_parts) > 1 else ""
            mean = ""
            standard_deviation = ""
            executor = ""
            for k, v in section["BenchmarkResults"].items():
                if k == "@name":
                    executor = v
                    continue
                if k == "mean":
                    mean = v["@value"]
                    continue
                if k == "standardDeviation":
                    standard_deviation = v["@value"]
                    continue
            records.append({
                "benchmark_name": benchmark_name,
                "dimension": dimension,
                "variant": variant,
                "executor": executor,
                "value_type": value_type,
                "size": size,
                "mean": mean,
                "standard_deviation": standard_deviation,
            })
    return records


def main():
    _, _, files = next(os.walk("."))
    for xml_file in files:
        if not xml_file.endswith(".xml"):
            continue
        try:
            with open(xml_file, "r") as fh:
                d = xmltodict.parse(fh.read())
                res = parse_xml_dict(d)
            with open(xml_file.replace(".xml", ".json"), "w") as outfile:
                json.dump(res, outfile)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
