# Sumotosima
(**Sum**ariser for **otos**copic **ima**ge) : A novel framework and dataset for classifying and generating summaries for otoscopic images of the middle ear, with the objective of developing summaries that are both well-defined and patient-friendly, addressing the challenge of insufficient explanations from medical professionals due to their hectic schedules and limited time per patient.
# Abstract
Otoscopy is a diagnostic procedure to examine the ear canal and eardrum using an otoscope. It identifies conditions like infections, foreign bodies, ear drum perforations and ear abnormalities. We propose a novel resource efficient deep learning and transformer based framework, Sumotosima (Summarizer for otoscopic images), an end-to-end pipeline for classification followed by summarization. Our framework works on combination of triplet and cross-entropy losses. Additionally, we use Knowledge Enhanced Multimodal BART whose input is fused textual and image embedding. The objective is to provide summaries that are well-suited for patients, ensuring clarity and efficiency in understanding otoscopic images. Given the lack of existing datasets, we have curated our own OCASD (Otoscopic Classification And Summary Dataset), which includes 500 images with 5 unique categories annotated with their class and summaries by Otolaryngologists. Sumotosima achieved a result of **98.03%**, which is **7.00%**, **3.10%**, **3.01%** higher than K-Nearest Neighbors, Random Forest and Support Vector Machines, respectively, in classification tasks. For summarization, Sumotosima outperformed GPT-4o and LLaVA by **88.53%** and **107.57%** in ROUGE scores, respectively. We have made our code and dataset publicly available
# Citation
If you find our codes or paper helpful, please consider citing.
@misc{khan2024sumotosimaframeworkdatasetclassifying,
      title={Sumotosima: A Framework and Dataset for Classifying and Summarizing Otoscopic Images}, 
      author={Eram Anwarul Khan and Anas Anwarul Haq Khan},
      year={2024},
      eprint={2408.06755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.06755}, 
}
