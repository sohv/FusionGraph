# Document Organization Guide


```
documents/
├── text/              
│   ├── reports/       
│   ├── manuals/       
│   └── articles/      
├── images/            
│   ├── diagrams/      
│   ├── screenshots/   
│   └── photos/        
└── examples/         
    ├── sample/        
    └── sample_ai/     
```

## How to Use

### Adding Your Documents

1. **Text Documents**: Place your PDF, TXT, DOCX files in `documents/text/`
   - Create subfolders by topic/project for better organization
   - Supported formats: PDF, TXT, DOCX, MD

2. **Images**: Place your PNG, JPG, etc. files in `documents/images/`
   - The Visual RAG system will process these for image-text queries
   - Supported formats: PNG, JPG, JPEG, GIF, BMP

### Streamlit App Configuration

The default paths in the webapp are now:
- **Document Directory**: `./documents/text`
- **Image Directory**: `./documents/images`

You can change these paths in the Streamlit sidebar if you prefer a different organization.

### Example Workflow

1. **Add your documents**: Copy your PDFs to `documents/text/`
2. **Add your images**: Copy your images to `documents/images/`
3. **Run the app**: Use the default paths or customize them
4. **Initialize system**: Click "Initialize System" in the sidebar
5. **Query**: Ask questions that can reference both text and images

### Git Considerations

The `documents/` folder structure is tracked in git, but you may want to add your actual content to `.gitignore`:

```
documents/text/*
documents/images/*
!documents/text/.gitkeep
!documents/images/.gitkeep
```

This keeps the folder structure but ignores your actual documents for privacy/size reasons.