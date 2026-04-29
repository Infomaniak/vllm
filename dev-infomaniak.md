# **Debugging vLLM with PyCharm and Docker**

Setting up PyCharm to debug vLLM inside a Docker container can be tricky due to two main issues:

1. **Missing C Extensions:** The live host source code mounted into the container lacks the compiled C extensions (vllm/\_C\*.so) built inside the image.  
2. **Module Shadowing (Circular Imports):** If paths are set up incorrectly, Python might mistakenly import local vLLM submodules (like vllm/tokenizers) instead of official PyPI packages (like HuggingFace's tokenizers), causing immediate crashes.

Follow this guide to configure your environment correctly.

## **1\. The Dockerfile "Symlink Hook" (Pre-requisite)**

Ensure your Dockerfile contains a .pth startup hook. Because PyCharm bind-mounts your local source code over the container's installed vLLM package, it overwrites the compiled .so files.

The hook runs at Python startup and automatically creates symlinks from the *installed* .so files into your *mounted* source code tree.

*(Note: If you are using the vLLM Dockerfile provided in this repository, this is already done for you).*

## **2\. Project Structure: Unmark "Sources Root"**

PyCharm tries to be helpful by automatically adding "Sources Roots" to your PYTHONPATH. If it adds the nested vllm folder, it will cause module shadowing (e.g., breaking HuggingFace transformers).

1. Open the **Project** tool window (usually on the left).  
2. Look at your project tree. Ensure the top-level folder (e.g., vllm-infomaniak) is the root.  
3. Find the nested vllm folder inside it.  
4. **Crucial:** If the nested vllm folder is colored **blue**, right-click it \-\> **Mark Directory as** \-\> **Unmark as Sources Root**. It should be the standard folder color.

## **3\. PyCharm Run/Debug Configuration**

Never run the entrypoint files as a "Script". You must run them as a "Module" from the project root.

1. Open **Run/Debug Configurations** in PyCharm.  
2. Create a new **Python** configuration.  
3. Change the execution type from **Script** to **Module** (using the dropdown next to the path input).  
4. Enter the module path. For the API server, use:  
   vllm.entrypoints.openai.api\_server  
5. Set the **Working directory** strictly to the project root (e.g., C:/Users/.../vllm-infomaniak). Do *not* set it to the nested vllm folder.  
6. Check your **Docker container settings** to ensure it's mapping the volume correctly (e.g., \-v C:/.../vllm-infomaniak:/opt/project).

## **Troubleshooting Common Errors**

### **Error: ImportError: cannot import name ... (most likely due to a circular import)**

**Symptom:** You see crashes deep inside HuggingFace transformers trying to load tokenizers or get\_config.

**Fix:** This is Module Shadowing. You are running as a script instead of a module, or you have the nested vllm directory marked as a blue "Sources Root" in PyCharm. Revisit Steps 2 and 3\.

### **Error: ModuleNotFoundError: No module named 'vllm.\_C'**

**Symptom:** Python can't find the compiled C++ components.

**Fix:** The symlink hook in the Dockerfile failed, or you rebuilt the container and the paths changed. Rebuild the Docker image to ensure the C-extensions are compiled, and verify the .pth setup script ran successfully.