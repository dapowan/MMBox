import json
import os
import random


from utils import save_json_to_log, find_objects, diff_list, read_yaml_file, save_dict_to_json, string_to_json, \
    load_json
from utils_llm import LLMProxy, generate_and_extract

QUERY_TYPES = ["Time", "Duration", "Frequency", "Existence", "Action", "Place", "Summary"]

def generate_tags(model, tags_seed, gen_tag_meta, config, log_path):

    def sample_tags_examples(tag_num, example_num):
        chosen_tags = random.sample(tags_seed, min(tag_num, len(tags_seed)))

        results = []
        for tag_obj in chosen_tags:
            examples = tag_obj.get("examples", [])
            chosen_examples = random.sample(examples, min(example_num, len(examples)))
            results.append({
                "tag": tag_obj["tag"],
                "description": tag_obj["description"],
                "examples": chosen_examples
            })
        return results

    prompt_values = {}
    tags_examples = sample_tags_examples(config["gen_tag_num"], config["gen_tag_example_num"])
    prompt_values["tag"] = tags_examples[0]["tag"]
    prompt_values["description"] = tags_examples[0]["description"]
    prompt_values["examples"] = str(tags_examples[0]["examples"])
    prompt_values["n_output"] = config["gen_tag_example_output"]

    tag_target = tags_examples[0]["tag"]
    print(f"Generating tags for {tag_target} with examples from: {tags_examples}")
    generated_results = generate_and_extract(model, gen_tag_meta["prompt_template"], prompt_values,
                                             gen_tag_meta["target_output"]["regex_extract_target_output"]["pattern"])
    tags_examples_generated = string_to_json(generated_results["extracted"][0])
    if tags_examples_generated:
        print(f"Generating tags examples for {tag_target}: {tags_examples_generated}")
        save_json_to_log(generated_results, os.path.join(log_path, "generated_outputs_tags.log"))
        tag_examples = find_objects(tags_seed, "tag", tag_target)[0]
        tags_examples_new = diff_list(tags_examples_generated, tag_examples["examples"])
        print(f"New tag examples insert for {tag_target}: {tags_examples_new}")
        tag_examples["examples"] = tag_examples["examples"] + tags_examples_new
    else:
        print("No tags examples were generated.")
    return tags_seed


def main(model, query_meta_seed, gen_tag_meta, gen_template_meta, config, output_dir):
    query_tags = query_meta_seed["tags"]
    for i in range(2):
        output_dir_i = os.path.join(output_dir, f"round_{i}")
        query_tags = generate_tags(model, query_tags, gen_tag_meta, config, output_dir_i)

        query_meta_seed["tags"] = query_tags


        save_dict_to_json(query_meta_seed, os.path.join(output_dir_i, "query_meta.json"))
    pass

if __name__ == '__main__':
    key = 'sk-BR9dBZyz2iF4VDfoA73aD6691f834cB5B4C6Ba33562cB9E8'
    name = 'gpt-4o-2024-08-06'
    model = LLMProxy(name, key)
    output_dir = "log/t0"
    config = {"gen_tag_num":1, "gen_tag_example_num": 5, "gen_tag_example_output": 10}
    query_meta_seed = load_json("dataset/query_meta.json")
    gen_tag_meta = read_yaml_file("prompt/gen_tag_example.yaml")
    main(model, query_meta_seed, gen_tag_meta, None, config, output_dir)