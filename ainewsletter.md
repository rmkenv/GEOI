# Generate just the FREE teaser (default)
python ainewsletter.py --api-key YOUR_KEY

# Generate everything at once
python ainewsletter.py --generate-all --api-key YOUR_KEY

# Generate premium full only
python ainewsletter.py --generate-premium --api-key YOUR_KEY

# Generate specific vertical deep-dive
python ainewsletter.py --generate-vertical "Satellite Imagery" --api-key YOUR_KEY

# Test with 10 stocks first
python ainewsletter.py --limit 10 --generate-all --api-key YOUR_KEY
```

## 📁 Output Structure
```
newsletters/
├── newsletter_free_overview.md              # FREE edition
├── newsletter_premium_full.md               # PREMIUM full
├── newsletter_premium_vertical_Satellite_Imagery.md
├── newsletter_premium_vertical_GIS_Software.md
├── newsletter_premium_vertical_Drone_Technology.md
└── ... (one for each vertical)
