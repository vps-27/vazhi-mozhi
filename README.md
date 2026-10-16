ிThe path from Tamil to English.

Vazhi Mozhi is a small web app that translates Tamil sentences into English. The translation model behind it is a Transformer that I built and trained from scratch in PyTorch, with no pretrained weights and no translation API. You type a Tamil sentence, press a button, and the model writes the English one.

The name means "the path of language": vazhi is path, mozhi is language.
<img width="956" height="512" alt="Screenshot 2026-10-16 193056" src="https://github.com/user-attachments/assets/8afa439b-ce61-462e-9d93-f3ce4e9c83b7" />
<img width="959" height="506" alt="Screenshot 2026-10-16 193110" src="https://github.com/user-attachments/assets/b9dd0b8f-43dd-428c-b087-746abf1e6239" />
Why I built this

I wanted to understand how machine translation really works, so I decided to build every piece myself instead of calling a ready-made model: the tokenizers, the attention layers, the training loop, and the interface on top.

I also wanted the interface to feel like it belongs to the language it translates. So the site has a Tamil heritage look (warm cream and terracotta colours, flower garlands, kolam patterns) and every section is written in both English and Tamil.

What it does
Translates a Tamil sentence into English with a Transformer encoder–decoder.
Has an intro page, a translator page, and five sections: About, How It Works, Examples, Tech Stack and Impact. Each one is in English and Tamil.
Lets you try sample Tamil sentences with one click.
Shows loading feedback and clear error messages if the model files are missing.
How it works
You enter a Tamil sentence.
A Byte Pair Encoding (BPE) tokenizer splits it into subword tokens.
The Transformer encoder reads all the Tamil tokens and builds a representation of the sentence.
The decoder writes English tokens one at a time. At each step it picks the most likely next token (greedy decoding) until it produces an end-of-sentence token.
The English tokenizer turns those tokens back into readable text.
The model
	
Architecture	Transformer encoder–decoder, written from scratch in PyTorch
Model dimension	256
Attention heads	4
Feed-forward size	1024
Layers	3 encoder, 3 decoder
Parameters	about 16.5 million
Tokenizers	Custom BPE, one for Tamil (10,949 tokens) and one for English (16,000 tokens)
Dataset	gopi30/english-tamil from Hugging Face
Training data	Up to 200,000 sentence pairs, cleaned and deduplicated
Decoding	Greedy

The attention, positional encoding, encoder and decoder layers are all my own implementation, following the design in the paper Attention Is All You Need (Vaswani et al., 2017).

Be honest about the quality

This is a learning project, and the model is small and trained on a limited amount of data. It handles some short, simple sentences reasonably and gets others wrong, sometimes badly. Please treat the output as an experiment, not as a reliable translation. The app says so on the page too.

If you want to push it further, the biggest improvements would come from more and cleaner training data, a longer training run, and a better decoding method such as beam search.

Run it on your computer

You need Python 3.10 or newer.

bash
git clone https://github.com/YOUR-USERNAME/vazhi-mozhi.git
cd vazhi-mozhi
pip install -r requirements.txt
python -m streamlit run app.py

Then open http://localhost:8501 in your browser.

On a normal laptop the CPU version of PyTorch is enough. The model is small, so translation takes about a second.

What's in the folder
app.py                     The Streamlit app: interface, model code and translation
best_model.pt              Trained model weights
tamil_tokenizer.json       Tamil BPE tokenizer
english_tokenizer.json     English BPE tokenizer
heritage_background.jpg    Background image for the interface
requirements.txt           Python dependencies
.streamlit/config.toml     Forces the light cream theme
train_vazhi_mozhi_colab.py Training script for Google Colab

app.py expects the model, the two tokenizers and the background image to sit next to it. If the image is missing, the app falls back to a plain gradient.

Train your own

train_vazhi_mozhi_colab.py runs the whole pipeline in one go: it downloads and cleans the data, trains both tokenizers, trains the Transformer, checks a few translations, and saves best_model.pt with the tokenizers.

Open a new notebook in Google Colab and choose a GPU runtime.
Run !pip install -q datasets tokenizers sacrebleu in the first cell.
Paste the script into the second cell and run it.
Download the zip it creates and copy the three files into this folder.

You can change the dataset size, model size and number of epochs at the top of the script. The app reads the model dimensions from the checkpoint, so a differently sized model loads without changing app.py.

Deploying

The app runs on Streamlit Community Cloud. Push this repository to GitHub, create an app from it, and set the main file to app.py. The included requirements.txt installs the CPU build of PyTorch, which keeps the install small.

Built with

Python, PyTorch, Streamlit, and the Hugging Face tokenizers and datasets libraries.

Acknowledgements
The gopi30/english-tamil dataset on Hugging Face.
Attention Is All You Need, Vaswani et al., 2017.
The fonts Playfair Display, Nunito Sans and Noto Sans Tamil from Google Fonts.
