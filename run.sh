# running script

python run_M2M.py \
-e './data/processed/Monkey/__Walk.bvh' \
-d cpu \
--source './data/processed/Flamingo/Flamingo_Walk.bvh' \
--mapping_file configs/mappings_flamingo.json \
--output_dir ./demo_output \
--sparse_retargeting \
--matching_alpha 0.9
