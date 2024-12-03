run_training() {
    local llm=$1
    local vision_tower=$2
    local task=$3
    local stages=$4
    local epochs=$5

    python llava/train/euclid_train.py \
        --model_name_or_path "$llm" \
        --vision_tower "$vision_tower" \
        --output_dir "./checkpoints/euclid-qwen-${llm}-${vision_tower}-${task}-${stages}" \
        --test_data_path playground/data/testing_data/data.json \
        --tasks "$task" \
        --stages "$stages" \
        --save_steps 2000 \
        --per_device_eval_batch_size 50 \
        --eval_steps 50 \
        --logging_steps 1 \
        --eval_accumulation_steps 1 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing False \
        --predict_with_generate True \
        --num_train_epochs "$epochs" \
        --attenuation_rate 0

    echo "Training for $llm $task $stages finished"

    # if we do not want to keep the trained model weights and training data
    rm -rf "./checkpoints/euclid-qwen-${llm}-${vision_tower}-${task}-${stages}"
    rm -rf "./playground/data/testing_data/training_buf/euclid-qwen-${llm}-${vision_tower}-${task}-${stages}"
}

run_training 0 "Qwen/Qwen2.5-1.5B-Instruct" "openai/clip-vit-large-patch14" "PointLiesOnLine_empirical" "1,2,3" "30"