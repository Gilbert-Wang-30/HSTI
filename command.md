cloning:

git clone git@github.com:Gilbert-Wang-30/HSTI.git


# Get latest changes from teammates
git pull origin main

# Edit files as needed...

# Save your changes
git add .
git commit -m "Added preprocessing script"
git push origin main


pcmci

python3 pcmci.py --start 0 --end 209 --lag 0
python3 heatmap.py --start 0 --end 209 --lag 0


python3 pcmci.py --start 1464 --end 1663 --lag 0
python3 heatmap.py --start 1464 --end 1663 --lag 0
