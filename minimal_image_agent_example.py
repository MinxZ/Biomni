"""
Example: Minimal Agent Configuration for Image Analysis Only

This example shows how to configure the Biomni agent to use only 
image analysis tools without default data lake and packages.
"""

from biomni.agent.a1 import A1


def create_minimal_image_agent():
    """Create an agent configured only for image analysis tasks."""
    
    # Initialize agent without downloading default data lake
    agent = A1(
        use_tool_retriever=True,
        download_data_lake=False, 
        llm='claude-sonnet-4-20250514'
    )
    
    # Clear default data lake to avoid distractions
    agent.data_lake_dict = {}
    
    # Keep only essential packages for image processing
    essential_packages = {
        'requests': 'HTTP library for downloading images from URLs',
        'base64': 'Encoding/decoding binary data',
        'PIL': 'Python Imaging Library for image processing',
        'cv2': 'OpenCV for computer vision tasks'
    }
    agent.library_content_dict = essential_packages
    
    # Filter module2api to keep only support tools (contains image function)
    image_modules = {
        'biomni.tool.support_tools': agent.module2api.get('biomni.tool.support_tools', [])
    }
    agent.module2api = image_modules
    
    print("✅ Configured minimal agent with:")
    print(f"   📦 {len(essential_packages)} essential packages")
    print(f"   🔧 {len(image_modules.get('biomni.tool.support_tools', []))} support tools (including image analysis)")
    print(f"   📊 {len(agent.data_lake_dict)} data lake items (empty)")
    
    return agent


def example_local_image_analysis():
    """Example using local image file analysis."""
    
    print("🚀 Local Image Analysis Example")
    print("="*50)
    
    # Create minimal agent
    agent = create_minimal_image_agent()
    
    # Add sample image descriptions (these could exist in your directory)
    agent.data_lake_dict = {
        'medical_scan.jpg': 'Medical imaging scan for diagnostic analysis',
        'research_chart.png': 'Scientific research data visualization',
        'lab_equipment.jpg': 'Laboratory equipment setup photo'
    }
    
    # Reconfigure after adding custom data
    agent.configure()
    
    log, result = agent.go(""" Can you use the image analysis tools to transform the url into text and still having the same structure 
    
    Based on the results described above in Examples 1 and 2, some of the modifications which showed the most pronounced effect on siRNA potency were applied to a siRNA duplex targeting the endogenous gene PTEN and screened in HeLa cells. Some of the siRNAs synthesized and tested are listed in Table 23.

### Table 23. siRNA Duplexes targeting PTEN and containing mismatch base pairings and nucleobase modifications in the sense strand

| Duplex ID | Strand | Sequence                                                                                                  | Modification | SEQ ID NO. |
|-----------|--------|-----------------------------------------------------------------------------------------------------------|--------------|------------|
| AD-19044  | S      | [Image](https://patentimages.storage.googleapis.com/6e/2d/78/6f0f5f0a84bad5/imgb0142.png)                  | parent       | 128        |
| AD-19044  | AS     | [Image](https://patentimages.storage.googleapis.com/6e/2d/78/6f0f5f0a84bad5/imgb0142.png)                  | parent       | 129        |
| AD-19045  | S      | [Image](https://patentimages.storage.googleapis.com/ef/11/9a/b6571dcc7b6645/imgb0143.png)                  | A9→G         | 130        |
| AD-19045  | AS     | [Image](https://patentimages.storage.googleapis.com/ef/11/9a/b6571dcc7b6645/imgb0143.png)                  | A9→G         | 129        |
| AD-19046  | S      | [Image](https://patentimages.storage.googleapis.com/c3/79/33/381e000f20cc71/imgb0144.png)                  | A9→C         | 131        |
| AD-19046  | AS     | [Image](https://patentimages.storage.googleapis.com/c3/79/33/381e000f20cc71/imgb0144.png)                  | A9→C         | 129        |
| AD-19047  | S      | [Image](https://patentimages.storage.googleapis.com/3a/aa/00/e6e8414265961e/imgb0145.png)                  | A9→U         | 132        |
| AD-19047  | AS     | [Image](https://patentimages.storage.googleapis.com/3a/aa/00/e6e8414265961e/imgb0145.png)                  | A9→U         | 129        |
| AD-19048  | S      | [Image](https://patentimages.storage.googleapis.com/3a/38/1a/92a9385c9be407/imgb0146.png)                  | C10→G        | 133        |
| AD-19048  | AS     | [Image](https://patentimages.storage.googleapis.com/3a/38/1a/92a9385c9be407/imgb0146.png)                  | C10→G        | 129        |
| AD-19049  | S      | [Image](https://patentimages.storage.googleapis.com/55/15/64/a8e8a38efafbc6/imgb0147.png)                  | C10→A        | 134        |
| AD-19049  | AS     | [Image](https://patentimages.storage.googleapis.com/55/15/64/a8e8a38efafbc6/imgb0147.png)                  | C10→A        | 129        |
| AD-19050  | S      | [Image](https://patentimages.storage.googleapis.com/e2/99/06/4b34fbbcf9856b/imgb0148.png)                  | C10→U        | 135        |
| AD-19050  | AS     | [Image](https://patentimages.storage.googleapis.com/e2/99/06/4b34fbbcf9856b/imgb0148.png)                  | C10→U        | 129        |
| AD-19051  | S      | [Image](https://patentimages.storage.googleapis.com/f4/ac/64/a3fb6c55ccd8a3/imgb0149.png)                  | A9→Y1        | 136        |
| AD-19051  | AS     | [Image](https://patentimages.storage.googleapis.com/f4/ac/64/a3fb6c55ccd8a3/imgb0149.png)                  | A9→Y1        | 129        |
| AD-19052  | S      | [Image](https://patentimages.storage.googleapis.com/28/d1/42/f91e1404ec7c1f/imgb0150.png)                  | C10→Y1       | 137        |
| AD-19052  | AS     | [Image](https://patentimages.storage.googleapis.com/28/d1/42/f91e1404ec7c1f/imgb0150.png)                  | C10→Y1       | 129        |
| AD-19053  | S      | [Image](https://patentimages.storage.googleapis.com/42/13/a8/6e1acd166d79fd/imgb0151.png)                  | C11→Y1       | 138        |
| AD-19053  | AS     | [Image](https://patentimages.storage.googleapis.com/42/13/a8/6e1acd166d79fd/imgb0151.png)                  | C11→Y1       | 129        |
| AD-19054  | S      | [Image](https://patentimages.storage.googleapis.com/dc/8d/53/b9ae3ab68fd9f9/imgb0152.png)                  | A12→Y1       | 139        |
| AD-19054  | AS     | [Image](https://patentimages.storage.googleapis.com/dc/8d/53/b9ae3ab68fd9f9/imgb0152.png)                  | A12→Y1       | 129        |
    """)
    
    print("📋 Agent Response:")
    print(result)
    return log, result


def example_url_image_analysis():
    """Example using image URL analysis with different modes."""
    
    print("🚀 URL Image Analysis Example")
    print("="*50)
    
    # Create minimal agent
    agent = create_minimal_image_agent()
    
    log, result = agent.go(""" Can you use the image analysis tools to transform the url into text and still having the same structure 
    
    Based on the results described above in Examples 1 and 2, some of the modifications which showed the most pronounced effect on siRNA potency were applied to a siRNA duplex targeting the endogenous gene PTEN and screened in HeLa cells. Some of the siRNAs synthesized and tested are listed in Table 23.

### Table 23. siRNA Duplexes targeting PTEN and containing mismatch base pairings and nucleobase modifications in the sense strand

| Duplex ID | Strand | Sequence                                                                                                  | Modification | SEQ ID NO. |
|-----------|--------|-----------------------------------------------------------------------------------------------------------|--------------|------------|
| AD-19044  | S      | [Image](https://patentimages.storage.googleapis.com/6e/2d/78/6f0f5f0a84bad5/imgb0142.png)                  | parent       | 128        |
| AD-19044  | AS     | [Image](https://patentimages.storage.googleapis.com/6e/2d/78/6f0f5f0a84bad5/imgb0142.png)                  | parent       | 129        |
| AD-19045  | S      | [Image](https://patentimages.storage.googleapis.com/ef/11/9a/b6571dcc7b6645/imgb0143.png)                  | A9→G         | 130        |
| AD-19045  | AS     | [Image](https://patentimages.storage.googleapis.com/ef/11/9a/b6571dcc7b6645/imgb0143.png)                  | A9→G         | 129        |
| AD-19046  | S      | [Image](https://patentimages.storage.googleapis.com/c3/79/33/381e000f20cc71/imgb0144.png)                  | A9→C         | 131        |
| AD-19046  | AS     | [Image](https://patentimages.storage.googleapis.com/c3/79/33/381e000f20cc71/imgb0144.png)                  | A9→C         | 129        |
| AD-19047  | S      | [Image](https://patentimages.storage.googleapis.com/3a/aa/00/e6e8414265961e/imgb0145.png)                  | A9→U         | 132        |
| AD-19047  | AS     | [Image](https://patentimages.storage.googleapis.com/3a/aa/00/e6e8414265961e/imgb0145.png)                  | A9→U         | 129        |
| AD-19048  | S      | [Image](https://patentimages.storage.googleapis.com/3a/38/1a/92a9385c9be407/imgb0146.png)                  | C10→G        | 133        |
| AD-19048  | AS     | [Image](https://patentimages.storage.googleapis.com/3a/38/1a/92a9385c9be407/imgb0146.png)                  | C10→G        | 129        |
| AD-19049  | S      | [Image](https://patentimages.storage.googleapis.com/55/15/64/a8e8a38efafbc6/imgb0147.png)                  | C10→A        | 134        |
| AD-19049  | AS     | [Image](https://patentimages.storage.googleapis.com/55/15/64/a8e8a38efafbc6/imgb0147.png)                  | C10→A        | 129        |
| AD-19050  | S      | [Image](https://patentimages.storage.googleapis.com/e2/99/06/4b34fbbcf9856b/imgb0148.png)                  | C10→U        | 135        |
| AD-19050  | AS     | [Image](https://patentimages.storage.googleapis.com/e2/99/06/4b34fbbcf9856b/imgb0148.png)                  | C10→U        | 129        |
| AD-19051  | S      | [Image](https://patentimages.storage.googleapis.com/f4/ac/64/a3fb6c55ccd8a3/imgb0149.png)                  | A9→Y1        | 136        |
| AD-19051  | AS     | [Image](https://patentimages.storage.googleapis.com/f4/ac/64/a3fb6c55ccd8a3/imgb0149.png)                  | A9→Y1        | 129        |
| AD-19052  | S      | [Image](https://patentimages.storage.googleapis.com/28/d1/42/f91e1404ec7c1f/imgb0150.png)                  | C10→Y1       | 137        |
| AD-19052  | AS     | [Image](https://patentimages.storage.googleapis.com/28/d1/42/f91e1404ec7c1f/imgb0150.png)                  | C10→Y1       | 129        |
| AD-19053  | S      | [Image](https://patentimages.storage.googleapis.com/42/13/a8/6e1acd166d79fd/imgb0151.png)                  | C11→Y1       | 138        |
| AD-19053  | AS     | [Image](https://patentimages.storage.googleapis.com/42/13/a8/6e1acd166d79fd/imgb0151.png)                  | C11→Y1       | 129        |
| AD-19054  | S      | [Image](https://patentimages.storage.googleapis.com/dc/8d/53/b9ae3ab68fd9f9/imgb0152.png)                  | A12→Y1       | 139        |
| AD-19054  | AS     | [Image](https://patentimages.storage.googleapis.com/dc/8d/53/b9ae3ab68fd9f9/imgb0152.png)                  | A12→Y1       | 129        |
    """)

    
    print("📋 Agent Response:")
    print(result)
    return log, result


def example_text_extraction():
    """Example using image for text extraction (OCR)."""
    
    print("🚀 Text Extraction Example")
    print("="*50)
    
    # Create minimal agent
    agent = create_minimal_image_agent()
    
    log, result = agent.go("""
    Please extract text from an image containing written content:
    
    1. Use this sample document image URL:
       https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Declaration_of_Independence_%28high_resolution%29.jpg/300px-Declaration_of_Independence_%28high_resolution%29.jpg
    
    2. Use mode="text_extraction" to extract all visible text
    3. Use model="claude-sonnet-4-20250514"
    4. Provide the extracted text in a readable format
    
    Use the read_and_summarize_image function for this OCR task.
    """)
    
    print("📋 Agent Response:")
    print(result)
    return log, result


if __name__ == "__main__":
    print("🧬 Minimal Biomni Agent - Image Analysis Only")
    print("="*60)
    
    # Run examples
    examples = [
        # ("Local Image Analysis", example_local_image_analysis),
        ("URL Image Analysis", example_url_image_analysis), 
        # ("Text Extraction (OCR)", example_text_extraction),
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n🔄 Running: {name}")
            log, result = example_func()
            if log and result:
                results[name] = {"status": "success"}
                print(f"✅ Completed: {name}")
            else:
                results[name] = {"status": "skipped"}
                print(f"⏭️ Skipped: {name}")
        except Exception as e:
            print(f"❌ Error in {name}: {str(e)}")
            results[name] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("📊 EXECUTION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status_icons = {"success": "✅", "failed": "❌", "skipped": "⏭️"}
        icon = status_icons.get(result["status"], "❓")
        print(f"{icon} {name}")
        if "error" in result:
            print(f"   └─ Error: {result['error']}")
    
    print("\n🔧 Image Function Usage:")
    print("   • read_and_summarize_image(image_source, mode, prompt, model)")
    print("   • Modes: 'general', 'scientific', 'medical', 'data_viz', 'text_extraction', 'custom'")
    print("   • Supports both local paths and URLs")
    print("   • Uses get_llm() for flexible model selection")
    print("   • Custom prompts override mode-based analysis")
    
    print("\n💡 Usage Pattern:")
    print("   ```python")
    print("   from biomni.tool.support_tools import read_and_summarize_image")
    print("   ")
    print("   # Local file")
    print("   result = read_and_summarize_image('./image.jpg', mode='scientific')")
    print("   ")
    print("   # URL with custom model")
    print("   result = read_and_summarize_image('https://...', mode='medical', model='gpt-4o')")
    print("   ")
    print("   # Custom prompt")
    print("   result = read_and_summarize_image(url, prompt='Identify all species in this image')")
    print("   ```")