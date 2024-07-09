import PyPDF2
import torch
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoProcessor, AutoModel
from pydub import AudioSegment
import bitsandbytes as bnb
import io
import scipy
from pydub import AudioSegment
from pydub.playback import play


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

os.environ["TOKENIZERS_PARALLELISM"] = "false"
class PDF_TO_PODCAST():

    def __init__(self) -> None:
        self.text_to_voicemodel = None
        self.summarizer_model = None
        self.document_text = [] #list of strings
        self.pipeline = None
        self.llama_model = "meta-llama/Llama-2-7b-chat-hf"
        self.access_token = "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"
        self.model_dir = "./llm_model"
        self.summary = None
        self.podcast_script = []
        if not os.path.exists(self.model_dir):
       
         #  self.initialise_model()
         # self.save_model()
        else:
     
           #self.load_model()
 
    def text_to_speech(self,text, processor, model, segment_name):
        inputs = processor(text=text, return_tensors="pt")
        outputs = model.generate(**inputs)
        sampling_rate = model.generation_config.sample_rate
        scipy.io.wavfile.write(segment_name, rate=sampling_rate, data=outputs.cpu().numpy().squeeze())
        
    def generate_podcast(self):
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = AutoModel.from_pretrained("suno/bark")
        #combined_audio = AudioSegment.empty()
        counter = 1
        for segment in self.podcast_script:
            audio = self.text_to_speech( segment, processor, model, segment_name=f"segment{counter}.wav")
            counter += 1


    def format_podcast_script(self):
 

        segment_1 = """
        [Segment 1: Understanding Mortgage Pads and the Attention Window]
        So, let's get started with the basics. When it comes to mortgages, padding is essential. You see, lenders need to ensure they can repay what they lend, and they do this by calculating your debt-to-income ratio. This is where the famous 29-41 rule comes into play. It's a quick way for lenders to assess your financial situation and determine if you can handle the mortgage payments. It's all about adding up your monthly expenses to get a realistic picture of your affordability.
        """

        segment_2 = """
        [Segment 2: Deciphering Mortgage Types and Lenders]
        Now, there are three main types of mortgage loans to choose from: conventional, big-lender, small-time, and government-backed. Each comes with its own set of pros and cons, so it's crucial to understand them. Conventional loans are typically offered by big banks, and they usually require a rather substantial down payment. Big-lender loans are a bit more flexible when it comes to down payment requirements, but they often come with higher interest rates. Small-time loans, as the name suggests, are offered by smaller lending institutions and may have more competitive interest rates. But be careful, as they might have less favorable terms or fewer benefits compared to the other options. Government-backed loans are often the most affordable and come with various programs to assist first-time homebuyers or those in specific industries.
        """

        segment_3 = """
        [Segment 3: Budgeting and Insurance]
        When considering a mortgage, budgeting is key. Lenders may require you to keep a certain balance on top of what you already owe. But don't despair; our handy budget calculator can help you calculate the true cost of each loan type, ensuring you don't take on more than you can handle. And speaking of handling finances, let's not forget about mortgage insurance. Most lenders offer private mortgage insurance, which protects them if you default on your loan. But it's not all bad; it can also protect you, ensuring you don't lose your home if you can't make payments due to financial hardship.
        """

        segment_4 = """
        [Segment 4: Planning and Research]
        Now, let's talk about planning. If you've never had a mortgage before, it's essential to do your research. Save up a healthy amount of money before even considering applying for a loan. Look into reputable lenders like Rydeco or Rocket Mortgage, and don't skimp on reviewing your credit scores and savings. Personal loans can also be a great option, especially for low-income earners through programs like Rydeco's. Understand the true cost of homeschooling with our personal lending program.
        """

        segment_5 = """
        [Segment 5: Closing Costs and Financial Setup]
        When it comes to closing costs, lenders will give you exact estimates based on their data. So, make sure you factor that into your calculations. And as you're doing all this financial legwork, remember to keep your financial situation in order. Lenders will want to see evidence of your income, assets, and creditworthiness. So, ensure you have all the necessary documentation ready when applying for a mortgage.
        """

        conclusion = """
        [Conclusion]
        In conclusion, navigating the mortgage process doesn't have to be daunting. By understanding the basics, doing your research, and staying organized with your finances, you can secure a great mortgage deal and move one step closer to your dream home. Remember to stay informed, and don't be afraid to seek help from professionals like our senior editor/producer, Rachel Richardson, who specializes in home-buying guidance. Until next time, happy huising!
        """

        self.podcast_script.append(segment_1)
        self.podcast_script.append(segment_2)
        self.podcast_script.append(segment_3)
        self.podcast_script.append(segment_4)
        self.podcast_script.append(segment_5)
        self.podcast_script.append(conclusion)
        
    def initialise_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model, use_auth_token=self.access_token)
        self.summarizer_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model, 
            use_auth_token=self.access_token
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.summarizer_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            #device=0,
        )

    def combine_multiple_audio_files(self):
        

        # List of WAV files to combine
        wav_files = ["bark_out.wav", "segment1.wav", "segment2.wav", "segment3.wav", "segment4.wav", "segment5.wav", "segment6.wav"]

        # Load the first audio file
        combined_audio = AudioSegment.from_wav(wav_files[0])

        # Iterate through the rest of the audio files and concatenate them
        for wav_file in wav_files[1:]:
            next_audio = AudioSegment.from_wav(wav_file)
            combined_audio += next_audio

        # Export the combined audio file
        combined_audio.export("combined_audio.wav", format="wav")

        print("Files combined successfully!")
    def summarization(self):
        context = "how large of a mortgage loan you can qualify for depends on how much debt a lender thinks you can take on as a borrower. This will ultimately determine how much house youre able to afford. Howeverjust because youre approved for a certain amount doesnt mean you should  with that home price. Instead, youll want to take a close look at your nancial health, including your household income and monthly expenses, and make sure to set a rm budget once you begin your home search. Read on to calculate how much house you can afford and learn what this means for . Home Affordability Calculator The Rocket Mortgage® Home Affordability Calculator gives you the option to see how much house you can afford, or how much cash you need for your down payment and closing costs. On the up side, if you have a price in mind, you can use a  to see how much cash you’ll need for a down payment and closing costs."
        #summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")
    
        #summary = summarizer(self.document_text[0][:4000], max_length = 2000,min_length =400, do_sample=False)
        hf_name = "pszemraj/led-base-book-summary"

        summarizer = transformers.pipeline(
            "summarization",
            hf_name,
            device=0 if torch.cuda.is_available() else -1,
        )
        result = summarizer(
            self.document_text[0],
            min_length=800,
            max_length=2000,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            repetition_penalty=3.5,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )
        self.summary = result[0]['summary_text']
        print(self.summary)
        return 0
    
    def generate_podcast_script(self):
        #self.summary = "When it comes to finding a mortgage that will pay you back the money you borrowed, lenders typically calculate your debt by dividing up your monthly debt and calculating how much extra debt you'll need to pay down. The 29-41 rule makes it even easier for lenders to figure out if you're really broke or if you just don't have the cash to pay it off. You only need to add up all your monthly expenses in order to get the best deal. This is where things get complicated. For instance, lenders might want to keep your balance on top of what you already owe, but they won't let you borrow more than that. Also, some lenders may charge you higher interest rates when you're refinancing. That's why we recommend using our handy budget calculator to find out exactly how much each loan type can cost. Then, use this calculator to make an educated decision about whether or not you can buy a new house. It also helps you know if there are any other loans that aren't too high-interest. Finally, consider considering your income before making a purchase. There are three types of mortgage loans available: conventional, big-lender, small-time, and government backed. Each one has its own set of downsides, such as time spent on the mortgage and property taxes. Remember, these are all part of the process of buying a home--you just need to stick with it long enough to be ready to start shopping. And remember, most lenders offer "private mortgage insurance," which means you won't have to worry about having to put so much money away every month. These are all great tips for getting started.1. Consider Your Home Price Deterioration Before Buying A Home Mortgage Every lender offers a variety of mortgage options, including conventional, large-loan, and small-business loans.2. Decide Who You Can Buy With In Which Bank If You Don'tHave Enough Money To Go On Sale Right NowIf you've never had a mortgage before, now's the time to plan. Make sure you have plenty of money saved up before you jump on the big-ticket item.3. Know Your Lending Options Purchase a home from a reputable lender like Rydeco or another company like Rocket Mortgage. Be sure to review your credit scores and savings before you go into buying. 4. Get a Personal Loan to Clerify Debt, Renovate Your Home and MoreRydeco offers a personal lending program designed specifically for low-income earners.5. Understand How Much Money You'll Need To Pay For Homebuying Most of the time, deciding how much space you'll save on mortgage payment depends on who you're spending your money on. All lenders will give you exact estimates of closing costs based on their real estate market data.6. Set Up Your Financial Dilemma When Shopping For A Home Is Done Everyone needs to know what kind of financing option to take advantage of when looking at open houses near you. Some lenders offer different types of mortgages, while others offer both conventional and big-lever style loans. After evaluating your financial situation, make sure everything is in line with your goals and objectives. 7. Review Your Finances & Sells My Personal Info Before You Apply For A Mortgage Because Of Her Skillful Writing and Story Editing, Miranda Crace graduated from the University of Wayne and is working as a creative writer for a short film studio. She loves to write and travel; she enjoys helping people achieve their goals through her writing and editing. 8. Get Help From Our Senior Editor/Producer Rachel Richardson About Us Covering Home Buying During Open House Sales Like Any Other Time, See What Happens Next Door Rentals Are Exceeded By Their Deductible Payment Amounts So many times over, you end up paying less than you thought you were going to. Even if you do manage to save money, you still shouldn't sell or share your personal information. We encourage you to contact us if you haven't yet."
        model_id = "CohereForAI/aya-23-8B"
        quantization_config = None
     
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        messages = [{"role": "user", "content": f"Generate a podcast script from the following content: {self.summary}"}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
       
        gen_tokens = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            do_sample=True, 
            temperature=0.3,
        )

        gen_text = tokenizer.decode(gen_tokens[0])
        print(gen_text)

        return 0
    
    def save_model(self):
        self.tokenizer.save_pretrained(self.model_dir)
        self.summarizer_model.save_pretrained(self.model_dir)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.summarizer_model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.summarizer_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            #device=0,
        )

    def parse_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
      
        self.document_text.append(text)
    


if __name__ == "__main__":

    pdf = PDF_TO_PODCAST()
    file_path = "./How Much House Can I Afford_.pdf"
    #pdf.parse_pdf(file_path)
    #pdf.summarization()
    #pdf.generate_podcast_script()
    #audio = AudioSegment.from_wav("bark_out.wav")
   # play(audio)
    #pdf.format_podcast_script()
    #pdf.generate_podcast()
    pdf.combine_multiple_audio_files()
