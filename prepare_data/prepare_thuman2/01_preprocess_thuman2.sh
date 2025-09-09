start=0
end=525

for file in $(seq -f "%04g" $start $end)
do
  echo $file
  echo "Process $(tput setaf 1)$file$(tput sgr0)"
  python prepare_data/prepare_thuman2/01_prepare_dataset.py --subject $file
done