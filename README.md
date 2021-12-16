# Capstone-Project4-Udacity-Data-Scientist
As part of the udacity.com Data Scientist nanodegree, this is project four and the final project of the program, otherwise known as our Capstone Project. This project involves Convolutional Neural Networks, including a web application that allows the user to upload an image of a dog or a human showcasing an AI that will output the closest dog breed resemblance to the image.

---
### Dog Breed Classification
Our model will process dog images:

<img src='images/sample_dog_output.png'/>

And also human images and predict the closest dog breed:

<img src='https://s3-prod.adage.com/s3fs-public/styles/800x600/public/20210809_libertyMutual_3x2.jpg'/>
Source: https://adage.com/article/hot-spots/watch-newest-commercials-directv-liberty-mutual-dodge-and-more/2356161

---
### Capstone Best Practices
I will be showcasing the following Data Science and Software Engineering Best Practices:

* Github and Code Quality including this repository along with comments, docstring in each function, class, or method, unit tests, and logical functions and PEP8 style guideline conventions
* ETL or Extract Transform Load data setup of a clean dataset
* Machine Learning including NLP techniques to process image data and the proper use of pipelines and grid search, traning vs. test data, and model evaluation
* Deployment of our web application to the IBM Cloud Foundry showing our Dog Breed Classification model and allowing users to upload images and receive dog breed recommendations
* Additional recommendation engine that recommends breeds based on Knowledge Based (qualities you are looking for in a dog) and Content Based (similarities to your features what other dog breeds resemble you, which will will hopefully introduce some serendipity or unexpected matches in our recommendations)

As in my previous project, I have also documented the work in the blog below:

#### My BLOG is HERE: https://hmelendez001.github.io/2022/01/26/Udacity-Data-Scientist-Nanodegree-Project-4-Capstone.html

# How to Run the Python Scripts
From the root folder run the following python command to process and clean the raw data:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Altenatively, to capture run times, from the root folder run the following python command with the time -v prefix:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; time -v python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

From the root folder run the following python command to train and save the model:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Altenatively, to capture run times, from the root folder run the following python command with the time -v prefix:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; time -v python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# How to Run the Web Application
From the app folder run the following python command:<p/>
##### &nbsp;&nbsp;&nbsp;&nbsp; python run.py
---
You should see something like this on a local browser
<img src='images/Project2DisasterDashboardBootstrap5.png'>
---
#### Alternatively you can run my IBM Cloud Foundry web installation here: https://helderdisasterdash.herokuapp.com/

From the dashboard landing page type a test message like **we need tents and water. We are in Silo, Thank you!** into the input text fields that reads _Enter a message to classify_. Click the "Classify Message" button and you should see something like the following. Notice it shows the categories for your message, if any: 

---
<img src='images/Project2DisasterResultBootstrap5.png'>
---

# Libraries Used
| Library | Description |
| :--- | :--- |
| Bootstrap | This toolkit was used to simplify writing the HTML5 elements including CSS styles in our web application |
| Flask | This Python micro web framework was used to write the backend of our web application dashboard |
| Gunicorn | This is needed for web application deployment |
| Heroku | This is used to access the hosting platform |
| JQuery | This is the javascript for manipulating objects on the UI web application dashboard |
| Json | This is the package for manipulating JSON text and objects |
| NLTK | This is used by the machine learning pipeline to do the text processing |
| Numpy | This is where numerical constants like np.nan came from|
| Pandas | This is the work horse behind our analysis for reading in the DataFrame from the CSV file and performing various data analysis and preparation |
| Pickle | This is the utility used to store the trained and optimized model for reuse by the UI web application dashboard |
| Plotly | This was the graphing library used to generate our visualizations |
| Re | This is the Regular Expression package used for manipulating strings with regex |
| Sklearn | This is the scikit-learn's Pipeline and GridSearchCV used to build and run our machine learning model |
| Sqlalchemy | This is the SQLite database package used to write and read to and from the SQL database |

# Files in this Repository
| File | Description |
| :--- | :--- |
| app | The dashboard or web application main directory |
| app > templates | Directory with the HTML files for the web application |
| app > templates > go.html | Page which displays the classification results of the web application |
| app > templates > master.html | The main page or the landing of the web application |
| app > \_\_init__.py | This file allows the folder to be set up as a Python module |
| app > run.py | The Flask file that runs the web application |
| data | The directory contaning the raw data for this project along with the scripts to run the ETL pipeline |
| data > DisasterResponse.db | The SQL Lite connection file |
| data > disaster_categories.csv | The raw data containing the disaster organization categories to process |
| data > disaster_messages.csv | The raw online messages to process |
| data > process_data.py | A data cleaning pipeline that loads the messages and categories datasets from the raw data, merges the two datasets, cleans the data, and stores it in a SQLite database using an SQLAlchemy engine |
| images | Used to store some screenshot images for this file |
| images > Project2DisasterDashboardBootstrap5.png | Screenshot of the web application landing page used below |
| images > Project2DisasterResultBootstrap5.png | Screenshot of the web application results page used below |
| models | The directory contaning the scripts to run the Text Processing, Natural Language Processing (NLP), and Machine Learning (ML) pipelines |
| models > MyTokenize.py | The file containing our tokenizer function MyTokenize so that we may reuse like a module |
| models > UrgencyWordExtractor.py | The file containing our UrgencyWordExtractor class so that we may reuse like a module |
| models > \_\_init__.py | This file allows the folder to be set up as a Python module |
| models > train_classifier.py | A machine learning pipeline that loads data from a SQLite database, splits the dataset into training and test datasets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, and exports the final model as a pickle file |
| Procfile | Instructs the runtime to use gunicorn to run our dashboard |
| README.md | The file you are currently reading |
| nltk.txt | This will tell Heroku to automatically download the modules in this file and install them into our application at build time |
| requirements.txt | Contains the list of required libraries, a subset of those listed in the "Libraries Used" section above but includes the versions required at run time |
| runtime.txt | The Python runtime version being used |

# How to Deploy to the Heroku Hosting Platform
| Step | Command | Description |
| :--- | :--- | :--- |
| 1 | mkdir web_app | Create a folder to contain all the assets to deploy |
| 2 | cp -R <root folder path of this project> web_app | Copy all the files in this project to the web_app folder you created in step 1 |
| 3 | heroku --version | Verify the Heroku installation | 
| 4 | curl https://cli-assets.heroku.com/install-ubuntu.sh \| sh | Install the necessary Heroku assets using their installation script | 
| 5 | heroku login -i | Assuming you have a Heroku account already, if not go to heroku.com and set up your account then issue this command to login with your credentials |
| 6 | git init<p/>git config --global user.email "you@example.com"<p/>git config --global user.name "Your Name" | Initialize a git repository with these ONE-TIME commands, if you do not already have one in Heroku |
| 7 | cd web_app<p/>cp -R app/templates .<p/>cp app/run.py . | You will need to copy the app/ directory contents to the root folder so that Heroku will find it at runtime **and** you will also have to update lines 16, 25, and 29 of run.py in the root folder to use models instead of ../models, data instead of ../data, and models instead of ../models |
| 8 | git add .<p/>git status<p/>git commit -m "your message" | Use this chain of command to commit your files to the Heroku git repository |
| 9 | heroku create my-app-name --buildpack heroku/python | Create a uniquely named Heroku app using this command. If you get a message that the app name is already taken, try again with a different app name until you find one that is not taken |
| 10| git remote -v | Check that heroku added a remote repository with this command |
| 11 | heroku config:set SLUGIFY_USES_TEXT_UNIDECODE=yes<p/>heroku config:set AIRFLOW_GPL_UNIDECODE=yes<p/>heroku config<p/>git push heroku master | Set any environment variable to pass along with the deployment and push the app to Heroku |
| 12 | From a web browser go to the link for your as named in step 8 above, something like https://app-name.heroku.com | Confirm your app is running on Heroku |

# Summary of the results
The dataset given was imbalanced (i.e. some labels like water have few examples and others like search_and_rescue, security, child_alone, shelter, clothing, etc. had none). We discovered this when first evaluating our model and seeing Scikit warnings that read "UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples." This imbalance affected training the model because our overall precision, recall, f1-score were skewed (with so many 0 results the averages were pulled down). Unlike with other data like financials, temperature readings there really is no way to necessarily impute the data. I cannot simply average out these gaps or even do other imputing strategies like fill forward or fill back data. NLP or Natural Language Processing does not give us these imputing options. Best we might do here to get a better evaluation result would be to emphasize the stats on the categories we know are not missing by passing the labels for the categories we do have.
  
Given more time I would have customized the web application dashboard further, using more of the fontawesome icons, adding some animation, and making the list of selected categories more reactive. Also, I would have included unit tests rather than running code from my Jupyter Notebook to test snippets. And finally I would have made the model pipeline component a module in PyPi.org to avoid having to depend on harcoding module paths or moving run.py from app to the root folder for Heroku deployments.

# Acknowledgements
Several code snippets came from previous lessons in our Udacity Data Scientist program. Also, where employed I have credited various contributors from StackOverflow.com, geeksforgeeks.org at https://www.geeksforgeeks.org/, https://www.tutorialspoint.com/ for sample plotly graphs, and the Data Science Stack Exchange at https://datascience.stackexchange.com. A big thank you to our instructors and all those involved in the Udacity program.

