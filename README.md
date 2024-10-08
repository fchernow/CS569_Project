# CS569_Project
The Effectiveness of Anti-Phishing Tools on Perturbed Logos Proposal
This repository contains the files and code used for the research project titled "The Effectiveness of Anti-Phishing Tools on Perturbed Logos". The project tests various anti-phishing tools by embedding clean and perturbed versions of common brand logos into basic HTML websites to evaluate the tools' detection accuracy.

Table of Contents
Project Overview
Repository Structure
Getting Started
Usage
Contributing
Team Members
License
Project Overview
Phishing attacks using perturbed logos are evolving to evade detection by security tools. This project aims to test anti-phishing tools such as Phish.AI and LogoGuard to determine their effectiveness in detecting logos that have been slightly modified (perturbed) using machine unlearning techniques.

The key research questions are:

Can imperceptible noise or perturbation added to logos bypass detection by anti-phishing tools?
Which anti-phishing tools are better equipped to detect these perturbed logos?
The project involves generating perturbed logos, embedding them in simple HTML sites, and testing these sites using different anti-phishing tools.

Repository Structure
graphql
Copy code
perturbed-logos/
│
├── images/                # Directory for clean and perturbed logos
│   ├── facebook_clean.png
│   ├── facebook_perturbed.png
│   ├── instagram_clean.png
│   ├── instagram_perturbed.png
│   ├── boa_clean.png
│   └── boa_perturbed.png
│
├── index.html             # Main HTML file to showcase clean and perturbed logos
│
├── README.md              # Documentation for the repository
│
└── scripts/               # Placeholder for additional scripts (if applicable)
Getting Started
Prerequisites
To run this project locally, you'll need:

A web browser (Chrome, Firefox, etc.)
Optional: Web server (if you want to host it locally)
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/perturbed-logos.git
Navigate to the project directory:
bash
Copy code
cd perturbed-logos
Open the index.html file in a web browser to view the clean and perturbed logos:
bash
Copy code
open index.html
Alternatively, you can set up a local web server if needed using Python:

bash
Copy code
python3 -m http.server
Then, open localhost:8000 in your web browser.

Usage
The index.html file displays clean and perturbed logos for testing with anti-phishing tools. You can use tools like Phish.AI and LogoGuard to assess the effectiveness of these tools in detecting the perturbed versions of the logos.

Adding New Logos
To add new logos:

Place the new logo images (both clean and perturbed versions) into the images/ directory.
Edit the index.html file to reference the new images.
Example:

html
Copy code
<div class="logo">
    <img src="images/yourlogo_clean.png" alt="Clean YourLogo">
    <p>Your Logo</p>
</div>

<div class="logo">
    <img src="images/yourlogo_perturbed.png" alt="Perturbed YourLogo">
    <p>Your Logo</p>
</div>
Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. You can also open an issue if you notice any bugs or have suggestions for improvements.

Team Members
Faith Chernowski: HTML development, website testing
Reagan Sanz: Machine unlearning and logo perturbation
Radhika Garg: Data analysis and documentation
