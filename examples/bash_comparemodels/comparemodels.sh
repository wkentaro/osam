#!/usr/bin/env bash

image_file='../_images/dogs.jpg'
number_of_choices=2

choices=$(samuel list | awk '{print $1}' | sed '1d')

echo "Select $number_of_choices models to compare:"
select model in $choices; do
    if [[ -n $model ]]; then
        echo "You have selected $model"
        selections+=("$model")
        ((COUNT++))
        if [[ $COUNT -eq $number_of_choices ]]; then
            break
        fi
    else
        echo "Invalid selection"
    fi
done

output_files=()
for model in "${selections[@]}"; do
    echo "Running $model"
    output_file=${model/:/-}.jpg
    samuel run $model --image $image_file | base64 --decode > $output_file
    echo "Exported $model.jpg"
    output_files+=($output_file)
done
echo "Compare results: ${output_files[@]}"
