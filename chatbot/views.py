import os
# import openai
from google.generativeai import configure, GenerativeModel
import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import ImageUpload, ChatHistory
from PIL import Image
import torch
import clip
import google.generativeai as genai

# Load CLIP model once
model, preprocess = clip.load("ViT-B/32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# genai.configure(api_key=os.getenv("your_key"))  //add your api key here

def home(request):
    """
    Show upload form and chat history (if image uploaded).
    """
    image_obj = None
    chats = []
    if request.method == "POST" and 'image' in request.FILES:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_obj = form.save()
            # Process image with CLIP to get description
            image_path = image_obj.image.path
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)

            labels = ["a dog", "a cat", "a person", "a car", "a tree", "a building","a Table"]
            text = clip.tokenize(labels).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
            similarities = (image_features @ text_features.T).squeeze(0)
            best_label = labels[similarities.argmax().item()]
            image_obj.description = best_label
            image_obj.save()
            chats = []
    else:
        form = ImageUploadForm()
    
    # If image_id is passed as GET param, load it and chats
    image_id = request.GET.get("image_id")
    if image_id:
        image_obj = get_object_or_404(ImageUpload, id=image_id)
        chats = ChatHistory.objects.filter(image=image_obj).order_by('timestamp')

    context = {
        'form': form,
        'image': image_obj,
        'chats': chats,
    }
    return render(request, 'chatbot/home.html', context)
@csrf_exempt
def chat_api(request):
    """
    AJAX endpoint to receive user message and return a manually generated bot response.
    Expects JSON with 'image_id' and 'message'.
    """
    if request.method == "POST":
        data = json.loads(request.body)
        image_id = data.get("image_id")
        user_message = data.get("message", "").strip().lower()
        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        image_obj = get_object_or_404(ImageUpload, id=image_id)

        # Generate prompt
        description = image_obj.description or "something"
        prompt = f"The image shows {description}. User asks: {user_message}"

        # Simple response logic based on description
        if "dog" in description:
            if "breed" in user_message:
                bot_reply = "It looks like a dog, but I can't identify the breed exactly."
            elif "doing" in user_message or "action" in user_message:
                bot_reply = "The dog seems to be posing for a picture."
            else:
                bot_reply = "That's a dog in the image. Ask me something about it!"
        
        elif "cat" in description:
            if "color" in user_message:
                bot_reply = "I can't say the exact color, but it's a cat!"
            else:
                bot_reply = "There's a cat in the image. What would you like to know about it?"

        elif "person" in description:
            bot_reply = "The image shows a person. You could ask about what they might be doing."

        elif "car" in description:
            bot_reply = "The image contains a car. Ask about its type, color, or use."

        elif "tree" in description:
            if "type" in description:
                bot_reply = "It's a tree. Maybe ask about the environment or type of tree"  
            else:
                bot_reply = "It could be an oak, maple, or sycamore tree."
           

        elif "building" in description:
            bot_reply = "This image shows a building. Want to ask about its purpose or structure?"

        elif "table" in description:
            bot_reply = "A table is visible. Maybe it's used for dining or working?"

        else:
            bot_reply = "I see something in the image, but can't say much about it. Ask a general question?"

        # Save to chat history
        ChatHistory.objects.create(image=image_obj, user_message=user_message, bot_response=bot_reply)

        return JsonResponse({"bot_reply": bot_reply})

    return JsonResponse({"error": "Invalid method"}, status=405)
