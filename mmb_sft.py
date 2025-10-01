# import some libraries
import os
import multiprocessing as mp

from evaluate import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, seed_everything, plot_images
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def main(model_id_or_path, dataset, output_dir, system, dataset_test=None):
    logger = get_logger()
    seed_everything(42)

    # ====== model ======


    # ====== dataset（本地）======
    data_seed = 42
    max_length = 2048
    split_dataset_ratio = 0.01  # 验证集比例
    num_proc = 1


    # ====== load local dataset ======
    # ✅ 直接把本地路径传给 load_dataset 即可；SWIFT 会自动识别 json/jsonl/csv/txt/目录
    train_dataset, val_dataset = load_dataset(
        dataset,
        split_dataset_ratio=split_dataset_ratio,
        num_proc=num_proc,
        seed=data_seed,                 # ✅ 删除了 model_name/model_author 的占位
    )

    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    logger.info(f'train_dataset[0]: {train_dataset[0]}')


    # ====== lora ======
    lora_rank = 8
    lora_alpha = 32
    #
    # # ====== training_args ======
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=16,
        num_train_epochs=10,
        metric_for_best_model='loss',
        save_total_limit=2,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    logger.info(f'output_dir: {output_dir}')
    #
    # # ====== model & template & lora ======
    model, tokenizer = get_model_tokenizer(model_id_or_path)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, tokenizer,
                            default_system=system, max_length=max_length)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')

    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')



    # ====== encode ======
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

    # 打印一条样本的模板化输入（可检查 EOS、role 拼接是否符合预期）
    template.print_inputs(train_dataset[0])

    # ====== train ======
    model.enable_input_require_grads()  # 兼容 gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

    images_dir = os.path.join(output_dir, 'images')
    logger.info(f'images_dir: {images_dir}')
    plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # save images


    if dataset_test is not None:
        evaluate(model_id_or_path, last_model_checkpoint, system, dataset_test, output_path=output_dir)

if __name__ == "__main__":
    mp.freeze_support()   # Windows 安全导入建议
    system = '''You are a helpful assistant that can analyze user's query and call correct tools to handle it.'''
    model_id_or_path = 'Qwen/Qwen3-8B-Base'
    output_dir = 'output/t0/'
    dataset_train = ['./dataset/v0_train.jsonl']  #
    dataset_test = ['./dataset/v0_test.jsonl']
    main(model_id_or_path, dataset_train, output_dir, system, dataset_test=dataset_test)

    # Visualize the training loss.
    # You can also use the TensorBoard visualization interface during training by entering
    # `tensorboard --logdir '{output_dir}/runs'` at the command line.


# Read and display the image.
# The light yellow line represents the actual loss value,
# while the yellow line represents the loss value smoothed with a smoothing factor of 0.9.
# from IPython.display import display
# from PIL import Image
# image = Image.open(os.path.join(images_dir, 'train_loss.png'))
# display(image)
