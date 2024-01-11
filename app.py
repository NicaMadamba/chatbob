# api/app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app, resources={r"/chatbot": {"origins": "https://computinginnovators.org/Student2/BSIT4C/Team_Rica/ccsict/index.php"}})

intents_data = [
        {
            "tag": "greetings",
            "patterns": [
                "Hi", "Hello", "Hey"
            ],
            "responses": [
                "Hi there, how can I help you?"
            ]
        },
        {
            "tag": "EnrollmentInfo",
            "patterns": [
                "How to enroll?",
                "How to enroll in BSIT?",
                "How to enroll in BSCS?",
                "How to enroll in BSCPE?",
                "How to enroll in BSECE?"
            ],
            "responses": [
                "Hi! let me know if you are an incoming freshman, transferee or upperclassman."
            ]
        },
        {
            "tag": "EnrollmentProcedures",
            "patterns": [
                "What are the procedures for enrollment for Freshmen and Transferees?", 
                "Freshmen or Freshman","Transferee", "incoming freshman",
                "How freshmen freshman transferee can enroll",
                "What are the steps or process in enrollment?",
                "What is the first step in enrolling",
                "Details on how can I enroll",
                "Where can I start with enrollment?",
                "How enrollment work?"
            ],
            "responses": [
                "The Enrollment Procedures is stated below:",
                "",
                "1.) Proceed to the concerned Program Chair/ Dean for interview;" , 
                "2.) Undergo Medical/ Dental Examination at the University infirmary;" , 
                "3.) Secure student number from the Office Of Student Affairs and Services (OSAS)/ Registrar's Office;" ,
                "4.) Submit admission requirements to the Registrars office where subjects are encoded and fees are assessed;" ,
                "5.) Pay the assessed fees at the Cashier's Office (as applicable);" ,
                "6.) For scholars, get approval of scholarship from the OSAS (as applicable);" , 
                "7.) Enroll National Service Training Program (NSTP) at the NSTP office (as applicable); and Proceed to the Campus Business Affair Office (CBAO) for I.D Processing."     
            ]
        },
        {
            "tag": "EnrollmentReqs",
            "patterns": [
                "What are the requirements for enrollment needed",
                "Enrollment requirements"
                
            ],
            "responses": [
                "Hi! These are the requirements for admission that will be submitted to the ISUC Registrar's Office: ",
                "",
                "Freshmen or Incoming Students:",
                "1.) Original copy of Grade 12 Report Card (form 138); ",
                "2.) Original copy of Certificate of Good Moral Character; ",
                "3.) Photocopy of senior high school diploma; ",
                "4.) Authenticated copy of PSA birth certificate; and ",
                "5.) Four (4) copies of a 2x2 ID picture (white background with nametag). ",
                "",
                "Transferee:",
                "1.) Original Copy of Certification of Grades showing all the subjects taken from the school last attended; ",
                "2.) Original Copy of Honorable Dismissal; ",
                "3.) Original Copy of Certificate of Good Moral Character; ",
                "4.) Four (4) copies of 2x2 ID picture (with white background and name tag); and ",
                "5.) Authenticated Copy of PSA Birth Certificate. ",
                "   ",
                "MasterClass:",
                "1.) Original Transcript of Records (TOR)",
                "2.) One PC of a 2x2 ID picture",
                "3.) GS Form No. 1 (Application for Admission Form)",
                "4.) GS Form No. 2 (Standard Recommendation Form)",
                "",
                "For more inquiries, email us at ccsict.isucabagan@isu.edu.ph."
            ]
        },
        {
            "tag": "FreshmanEnroll",
            "patterns": [
                "What are the requirements needed for Freshman or incoming students to enroll?", 
                "What are the required papers in admission",
                "Details on how a freshman can enroll"
                
            ],
            "responses": [
                "Hi! These are the requirements for admission that will be submitted to the ISUC Registrar's Office: ",
                "",
                "1.) Original copy of Grade 12 Report Card (form 138); ",
                "2.) Original copy of Certificate of Good Moral Character; ",
                "3.) Photocopy of senior high school diploma; ",
                "4.) Authenticated copy of PSA birth certificate; and ",
                "5.) Four (4) copies of a 2x2 ID picture (white background with nametag). "
            ]
        },
        {
            "tag": "TransfereeEnroll",
            "patterns": [
                "What are the requirements needed for Transferee to enroll?",
                "Details on how transferee can enroll"
            ],
            "responses": [
                "Here are the requirements for Transferees that you must submit to the ISUC Registrar's Office: ",
                "1.) Original Copy of Certification of Grades showing all the subjects taken from the school last attended; ",
                "2.) Original Copy of Honorable Dismissal; ",
                "3.) Original Copy of Certificate of Good Moral Character; ",
                "4.) Four (4) copies of 2x2 ID picture (with white background and name tag); and ",
                "5.) Authenticated Copy of PSA Birth Certificate. ",
                "   ",
                "   Also, please save these important dates for enrollment: the first semester is August 14 to September 8, and the second semester is February 12 to 16, 2024. Visit the official CCSICT Facebook page for detailed schedules."
            ]
        },
        {
            "tag": "UpperclassmanEnroll",
            "patterns": [
                "I'm an upperclassman", "Upper class man", "I'm an upper classman", "upperclassmen", "upperclass", 
                "How an upperclassman can enroll",
                "How old students can enroll?",
                "how sophomore, junior and senior students can enroll?"
            ],
            "responses": [
                "Hi! If you are a regular student, click the SARIAS link to enroll >> 'https://app.isucabagan.edu.ph/sarias/index.php'. ",
                "However, if you are a shifter, irregular, or returning student, go to the campus for enrollment. Submit your Pre-registration Form signed by your registration adviser at the Registrar's Office."
            ]
        },
        {
            "tag": "MasterClass",
            "patterns": [
                "How can I enroll in Master class?", 
                "How can i enroll in your masterclass?",
                "How to enroll Masteral",
                "How to enroll in graduate school",
                "How to enroll in Master of Science in Information Technology",
                "How to enroll in Postgraduate Diploma in Data Science and Analytics"
            ],
            "responses": [
                "Here are the requirements for admission to graduate school:",
                "",
                "1.) Original Transcript of Records (TOR)",
                "2.) One PC of a 2x2 ID picture",
                "3.) GS Form No. 1 (Application for Admission Form)",
                "4.) GS Form No. 2 (Standard Recommendation Form)",
                "",
                "For more inquiries, email us at ccsict.isucabagan@isu.edu.ph.",
                "",
                "Also, please take note of these important dates for admission for graduate students:",
                "First Semester: 3rd week of May- 4th week of July",
                "Second Semester: 1st week of December- 2nd week of January"
            ]
        },
        {
            "tag": "AdmissionReqs",
            "patterns": [
                "What are the admission requirements?",
                "What documents do I need to submit with my application?", 
                "What are the admission requirements for undergraduate programs?"
            ],
            "responses": [
                "Hi! Here are the requirements for admission:",
                "1.) Original copy of Grade 12 Report Card (form 138);",
                "2.) Original copy of Certificate of Good Moral Character; photocopy of senior high school diploma;",
                "3.) Authenticated copy of PSA birth certificate; and ",
                "4.) Four (4) copies of a 2x2 ID picture (white background with nametag). ",
                "The requirements stated above will be submitted to the ISUC Registrar's Office."
            ]
        },
        {
            "tag": "RegProcedures",
            "patterns": [
                "What is the procedures for registration?", 
                "how to register for admission or enrollment"
            ],
            "responses": [
                "The Registration Procedures for Admission is stated below:",
                "",
                "1.) Accomplish application form for Admision test from the Office of Student Affairs and Services (OSAS) through the guidance office." , 
                "2.) Pay testing fee at the Cashier's office (as applicable); and" , 
                "3.) Take the test and secure test result's from the Guidance unit" 
            ]
        },
        {
            "tag": "DropingProcedure",
            "patterns": [
                "how to Drop Subjects?", "What are the Dropping of Subjects Procedure?"
            ],
            "responses": [
                "The Dropping of Subject Procedures is stated below:",
                "1.) Secure a dropping form from the Registrar's Office;" ,
                "2.) Accomplish the dropping form to be signed by th subject Instructor/Professor and the Registration Adviser noteb by the Dean/Program Chair; and" ,
                "3.) Submit a copy of the form at the Registrar's Office one week after the last day of enrollment in a term." ,
                "",
                "NOTE Any students who fails to attend classes shall be cosidered dropped. Subjects officially dropped within (3) days after the start of classes will no longer be reflected in the TOR. "
            ]
        },
        {
            "tag": "AddingProc",
            "patterns": [
                "How to Add Subjects?", 
                "how to add/change subjects?",
                "What are the Adding of Subjects Procedure?"
            ],
            "responses": [
                "The Adding of Subject Procedures is stated below:",
                "",
                "1.) Secure adding form from the Registrar's office;" , 
                "2.) Accomplish adding form to be signed by the subject Instructor/Proffesor and the Registration Adviser; noted by the Dean/Program Chair and for approval by the registrar;" ,
                "3.) Pay adding fee to the Cashier's Office (as applicable) and" ,
                "4.) Submit approved adding form to the Registrar's Office within 7 days after the first day of the classes."
            ]
        },
        {
            "tag": "CrossEnroll",
            "patterns": [
                "How does the Cross Enrollment works?", "How to cross enroll?"
            ],
            "responses": [
                "1.) For student who will cross enroll within the system and to other Higher Education Institution (HEIs):" , 
                "2.) Seek recommendation from the Program/ Department Chair, Dean and Registrar, and approval from the Executive Officer/Campus Administrator." ,
                "3.) For outside students wh will cross enroll within the University:" ,
                "4.) Present to the OSAS the Permission to Cross Enroll Form secured from their present school;" ,
                "5.) Submit to the Registrar the Permission to Cross Enroll Form recognized by the OSAS; and" , 
                "6.) At the end of the semester the student will be issued a COG after completion of the subject."
            ]
        },
        {
            "tag": "Shift",
            "patterns": [
                "How to shift of major field or program?"
            ],
            "responses": [
            "Here is the guidelines you need before you can Shift into another Major Field or Profram:",
            "",
            "1.) Have completed at least one semester in the program;" , 
            "2.) Secure, accomplish and submit duly approved shifting form to the Registrar's Office, and" , 
            "3.) Be allowed to shift twice only subject to the policies of the admitting college/ department and the grade requirement of the program."
            ]
        },
        {
            "tag": "GradingSystem",
            "patterns": [
                "What is the Grade Equivalents?", "Grading system"
            ],
            "responses": [
                "The following approved grading system shall be adopted",
                "",
                "Percent Equivalent---------Grade-------Description",
                "98-100---------------------1.00---------Excellent" , 
                "95-97----------------------1.25---------Very Satisfactory" ,
                "92-94----------------------1.50---------Satisfactory" ,
                "89-91----------------------1.75---------Fairly Satisfactory" , 
                "86-88----------------------2.00---------Good" , 
                "83-85----------------------2.25---------Fairly Good", 
                "80-82----------------------2.50---------Fair", 
                "77-79----------------------2.75---------Below Fair" , 
                "75-76----------------------3.00---------Passed" , 
                "Incomplete-----------------INC----------Requirements not fully met" , 
                "74 and below---------------5.00---------Failed" , 
                "", 
                "A.) A grade of 5.00 means failed; re-enrollment of the subject is required. " , 
                "B.) An INC grade is given to a student whose class standing throughout the semester is passing but fails to satisfy any of the prescribed requirements by the subject teacher." , 
                "C.) Students who incurred Incomplete grades after the issuance of Honorable Dismissal will no longer be allowed to complete even if the reglementary period of one (1) academic year for the completion has not yet lapsed." ,
                "D.) For a student to be able to clear his/her deficiencies, should be officially enrolled in the University." ,
                "E.) Completion form shall be made within one academic year otherwise the Inncomplete mark shall automatically become 5.00" ,
                "F.) Completion form shall be accomplished and filed at the Registar Office." ,
                "G.) Completion fee of php. 50.00 per subject shall be paid at the Cashier's Office" ,
                "H.) Incomplete mark will no longer be reflected on the TOR if completed and accomplished withih the duration of one academic year."
            ]
        },
        {
            "tag": "AcadScholar",
            "patterns": [
                "What is an university scholar?"
            ],
            "responses": [
                "A University Scholar is a student carrying at least 15 academic load required in his college who obtained a GWA of at least 1.50."
            ]
        },
        {
            "tag": "ColScholar",
            "patterns": [
                "What is a College Scholar?"
            ],
            "responses": [
                "A College Scholar is a student carrying at least 15 academic load required in his college who obtained a GWA of at least 1.75."

            ]
        },
        {
            "tag": "TuitionFee",
            "patterns": [
                "Is there an application fee, and how can I pay it?",
                "What is the tuition fee, and what are the payment options?"
            ],
            "responses": [
                "Hi! Isabela State University-Cabagan Campus is a free-tuition university. This is by RA 10913 known as the 'Universal Access to Quality Tertiary Education Act of 2017'. In compliance with Section 4 of the Act, all Filipino students who are either currently enrolled at the time of its effectivity, or shall enrolled at any time thereafter, any avail of the exemption from paying tuition and other school fees units enrolled, in courses leading to a bachelor's degree in any SUC and LUC."
            ]
        },
        {
            "tag": "Scholarship",
            "patterns": [
                "Are there any scholarships available, and how can I apply for them?"
            ],
            "responses": [
                "Hi! There is a merit scholarship for the ISU-Cabagan campus, where university scholars should have at least a grade of 1.00 to 1.50 (92% to100%) and shall receive a 3,000 peso cash incentive. College Scholars should have at least 1.51 to 1.75% (89 to 91%) and shall receive a 2,000 peso cash incentive. \n Other scholarship programs will be posted on the ISUC official page."
            ]
        },
        {
            "tag": "WaitingListinAdmission",
            "patterns": [
                "Is there a waiting list for admission, and how does it work?"
            ],
            "responses": [
                "Yes, there is! Just visit the official page of the ISU-Cabagan campus to keep in touch. 'https://www.facebook.com/ISUCabaganAdmissionandGuidanceServices'"
            ]
        },
        {
            "tag": "OnlineEnrollment",
            "patterns": [
                "Is online registration available, and how can I access it?", "Is online enrollment is available?"
            ],
            "responses": [
                "Hi! Yes, online registration/enrollment is available. Here is the link 'https://app.isucabagan.edu.ph'. ",
                "   ",
                "(Note: Online enrollment is NOT applicable for shifters, irregular and returning students. Go to campus for enrollment and submit your Pre-registration Form signed by your registration adviser at the Registrar's Office.)"
            ]
        },
        {
            "tag": "GradReq",
            "patterns": [
                "What are the requirements for graduation?"
            ],
            "responses": [
                "Hi! The following below are the requiements for Graduation",
                "- A candidate shall apply for graduation at the Registrar's Office through the College Secretary four(4) weeks after the first day of classes during his/her last semester.",
                "- The candidate for graduation shall satisfactorily complete his/her deficiency/ies before the colege/campus academic council meeting.",
                "- For transferees, the residency requirements shall be atleast one (1) academic year prior to gradutaion."
            ]
        },
        {
            "tag": "EnrollmentDate",
            "patterns": [
                "When is enrollment?",
                "When is date for enrollment?", 
                "when is the date for enrollment for first semester and second semester", 
                "date of enrollment for undergraduate and graduates", 
                "Date for enrollment for incoming students or freshmen, transferees, upperclassmen, old students, and graduates"
            ],
            "responses": [
                "Hi! Here are the dates for the Enrollment for both Undergraduate and Graduate: ",
                "",
                "First Semester:",
                "----Undergraduate(New Entrants): 2nd week of August- 2nd week of September",
                "----Undergraduate: 3rd week of August- 2nd of September; ",
                "----Graduate: 4th week of August-  2nd week of September",
                "",
                "Second Semester:",
                "----Undergraduate: 3rd week of February;",
                "----Graduate: 3rd-4th week of February ",
                "",              
                "Visit the official CCSICT Facebook page for detailed schedules of enrollment"
            ]
        },
        {
            "tag": "FoundationDate",
            "patterns": [
                "When is the Foundation Day?", "foundation date"
            ],
            "responses": [
                "The Foundation day is normally held on 2nd week of June."
            ]
        },
        {
            "tag": "ContactAdmin",
            "patterns": [
                "How can I contact the support team or administrator",
                "is there a customer service support representative available?",
                "Can i speak to a real person?",
                "I want to talk to someone from support",
                "Transfer me to a human representative"
            ],
            "responses": [
                "I appreciate you preference for human assistance. For personalized support, please feel free to reach out to our email directly at isuc_ccsict@edu.ph. We're here to help with any questions or corncerns you have!"
            ]
        },
        {
            "tag": "CoursesOffered",
            "patterns": [
                "what are the ccsict programs?",
                "what are the courses offered in CCSICT?", 
                "What are the programs offered?", 
                "What are the programs present here"
            ],
            "responses": [
                "Hi! CCSICT offers Bachelor of Science in Computer Science (BSCS), Bachelor of Science in Information Technology (BSIT), Bachelor of Science in Computer Engineering (BSCpE), and Bachelor of Science in Electronics and Communications Engineering (BS-ECE)",
                "",
                "CCSICT is also now offering Master of Science in Information Technology, and Postgraduate Diploma in Data Science and Analytics."
            ]
        },
        {
            "tag": "SariasLink",
            "patterns": [
                "What is the link for SARIAS?"
            ],
            "responses": [
                "Hi! Here is the link for SARIAS 'https://app.isucabagan.edu.ph/sarias/index.php'"
            ]
        },
        {
            "tag": "SediLink",
            "patterns": [
                "What is the link for SeDi?"
            ],
            "responses": [
                "Hi! Here is the link for SeDi 'https://sedi.isucabagan.edu.ph/sedi/login/index.php'"
            ]
        },
        {
            "tag": "LocationCCSICT",
            "patterns": [
                "where is CCSICT?",
                "Where can I find CCSICT?","Where is the CCSICT location", "Where is your location?", "where does the College of Computing Studies, Information and Communication Technology located"
            ],
            "responses": [
                "The Isabela States Univeristy- Cabagan Campus' College of Computing Studies, Information and Communication Technology is located at Catabayungan, Cabagan, Isabela"
            ]
        },
        {
            "tag": "LocationISUC",
            "patterns": [
                "Where can I find Isabela State University- Cabagan Campus (ISUC)?"
            ],
            "responses": [
                "The Isabela States Univeristy- Cabagan Campus is located at Garita, Cabagan, Isabela."
            ]
        },
        {
            "tag": "LocationREGISTRAR",
            "patterns": [
                "Where can I find ISUC's Registrar Office?"
            ],
            "responses": [
                "The Isabela States Univeristy- Cabagan Campus' Registrar Office is located at Administration Building, Garita, Cabagan, Isabela."
            ]
        },
        {
            "tag": "Dean",
            "patterns": [
                "Who is the Dean?"
            ],
            "responses": [
                "The CCSICT Dean is Dr. Ivy M. Tarun."
            ]
        },
        {
            "tag": "BSCSPChair",
            "patterns": [
                "Who is the Program Chair of BSCS", "Who is the program chair of Bachelor of Science in Computer Science (BSCS)?"
            ],
            "responses": [
                "The Program Chair of Bachelor of Science in Computer Science (BSCS) is Dr. Amy Lyn M. Maddalora"
            ]
        },
        {
            "tag": "BSECEPChair",
            "patterns": [
                "Who is the Program Chair of Bachelor of Science in Electronics and Communications Engineering (BS-ECE)", "Who is the Program Chair of BS-ECE"
            ],
            "responses": [
                "The Program Chair of Bachelor of Science in Electronics and Communications Engineering (BS-ECE) is Engr. Jolan B. Sy"
            ]
        },
        {
            "tag": "BSCPEPChair",
            "patterns": [
                "Who is the Program Chair of Bachelor of Science in Computer Engineering (BSCpE)?", "Who is the Program Chair of BSCpE"
            ],
            "responses": [
                "The Program Chair of Bachelor of Science in Computer Engineering (BSCpE) is Ma. Christina V. Magabilin."
            ]
        },
        {
            "tag": "BSITPChair",
            "patterns": [
                "Who is the Program Chair of Bachelor of Science in Information Technology (BSIT)?", "Who is the Program of BSIT"
            ],
            "responses": [
                "The Program Chair of Bachelor of Science in Information Technology (BSIT) is Dr. Heherson B. Albano."
            ]
        },
        {
            "tag": "IntramsDate",
            "patterns": [
                "When is the Intramurals date?", "intrams date"
            ],
            "responses": [
                "Hi! The Cluster/Campus Intrams are usually held during the 4th week of October."
            ]
        },
        {
            "tag": "UniGamesDate",
            "patterns": [
                "When is the University Games date?"
            ],
            "responses": [
                "Hi! The University games are usually held during the 2nd week of December."
            ]
        },
        {
            "tag": "OrientationDate",
            "patterns": [
                "When is the Orientation day?", "orientation date"
            ],
            "responses": [
                "Hi! The Orientation Day is normally held a week before the start of classes."
            ]
        },
        {
            "tag": "WhereAcademicCalendar",
            "patterns": [
                "give me the school or university academic calendar of activities",
                "Where can I find the academic calendar?"
            ],
            "responses": [
                "Unfortunately, I dont have an ecopy of the University's Academic Calendar of Activities. However, you can check it out on the Bulletin Board located at CCSICT. Alternatively, if you're looking for a specific event, feel free to ask me and I'll try my best to find out the date for you."
            ]
        },
        {
            "tag": "ClubRegistration",
            "patterns": [
                "How can I get involved in student clubs and organizations?"
            ],
            "responses": [
                "Hi! You can register at your desired club during the orientation day, or you can approach your classroom president to assist you with your registration."
            ]
        },
        {
            "tag": "Clubs",
            "patterns": [
                "What are the student clubs present in CCSICT?"
            ],
            "responses": [
                "Hi! CCSICT offered clubs such as the Student Publication Club, Programming Club, Music Club, Dance Club, and Sports Club."
            ]
        },
        {
            "tag": "GraduationDate",
            "patterns": [
                "Is there a graduation ceremony, and when does it take place?"
            ],
            "responses": [
                "Hi! Graduation and Commencement exercises will take place anywhere between last week of July and 2nd week of August."
            ]
        },
        {
            "tag": "SeminarsOffered",
            "patterns": [
                "Are there workshops or seminars on campus to enhance skills and knowledge?"
            ],
            "responses": [
                "Yes, there are! Faculties often announce to the students if there are upcoming workshops and seminars. Just stay tuned."
            ]
        },
        {
            "tag": "PrelimDate",
            "patterns": [
                "When is the Preliminary Exam?",
                "When is the Undergraduate and Graduate Preliminary examination",
                "when will be the Date for preliminary examination for incoming students or freshmen, transferees, upperclassmen, old students, undergraduates, sophomore, junior, senior, and graduates start"

            ],
            "responses": [
                "Hi! Here are the dates for the Preliminary Examination:",
                "",
                "First Semester:",
                "----Undergraduate: 3rd week of October; ",
                "----Graduate: 4th week of October ",
                "",
                "Second Semester:",
                "----Undergraduate: Last week of March;",
                "----Graduate: 1st week of April "            
            ]
        },
        {
            "tag": "MidtermDate",
            "patterns": [
                "When is the Midterm Exam?", 
                "When is the Undergraduate and Graduate midterm examination",
                "when will be the Date for midterm examination for incoming students or freshmen, transferees, upperclassmen, old students, undergraduates, sophomore, junior, senior, and graduates start"

            ],
            "responses": [
                "Hi! Here are the dates for the Mid-term Examination:",
                "",
                "First Semester:",
                "----Undergraduate: 2nd week of December ",
                "----Graduate: 2nd week of December",
                "",
                "Second Semester:",
                "----Undergraduate: 2nd week of May",
                "----Graduate: 2nd week of May",
                "",
                "Midyear:",
                "----Undergraduate: Last week of July to 1st week of August",
                "----Graduate: 1st week of August"
            ]
        },
        {
            "tag": "FinalsDate",
            "patterns": [
                "When is the Final Exam?",
                "When is the Undergraduate and Graduate final examination",
                "when will be the Date for final examination for incoming students or freshmen, transferees, upperclassmen, old students, undergraduates, sophomore, junior, senior, and graduates start"
            ],
            "responses": [
                "Here are the Final Examination date for GRADUATING STUDENTS:",
                "",
                "First Semester:",
                "----Undergraduate: 3rd week of January ",
                "",
                "Second Semester:",
                "----Undergraduate: 2nd week of June",
                "",
                "Midyear:",
                "----Undergraduate: 3rd week of August",
                "","",
                "Here are the Final Examination date for NON-GRADUATING STUDENTS:",
                "",
                "First Semester:",
                "----Undergraduate: 4th week of January ",
                "----Graduate: 4th week of January",
                "",
                "Second Semester:",
                "----Undergraduate: 4th week of June",
                "----Graduate: 4th week of June",
                "",
                "Midyear:",
                "----Undergraduate: 4th week of August",
                "----Graduate: 4th week of August"
            ]
        },
        {
            "tag": "AdmissionDate",
            "patterns": [
                "When is the admission for new entrants?", "admission for new entants date", 
                "admission for graduate students date",
                "when will the Date for admission for incoming students or freshmen, transferees, upperclassmen, old students, undergraduates, sophomore, junior, senior, and graduates start"

            ],
            "responses": [
                "Hi! Here are the dates for the Admission for New Entrants:",
                "",
                "First Semester:",
                "----Undergraduate: 1st week of March- Last week of July",
                "----Graduate: 3rd week of May- Last week of July",
                "",
                "Second Semester:",
                "----Undergraduate: 2nd week of December- 2nd week of January",
                "----Graduate: 1st week of December- 2nd week of January"          ]
        },
        {
            "tag": "StartofClassesDate",
            "patterns": [
                "When is the start of classes?", "start of classes date",
                "Start of classes for first semester, second semester, midyear",
                "start of classes for incoming students or freshmen, transferees, upperclassmen, old students, undergraduates, sophomore, junior, senior, and graduates start",
                "starting date of classes for 1st year, 2nd year, 3rd year, 4th year, masterclass, and master class"

            ],
            "responses": [
                "Hi! Here are the Start of Classes dates:",
                "",
                "First semester:",
                "----Undergraduate: 3rd week of September",
                "----Graduate: 3rd week of September",
                "",
                "Second semester:",
                "----Undergraduate: 3rd week of February",
                "----Graduate: 4th week of Febuary",
                "",
                "Midyear:",
                "----Undergraduate: 3rd week of July",
                "----Graduate: 3rd week of July"
            ]
        },
        {
            "tag": "AddingSubjectDate",
            "patterns": [
                "When is the adding or changing of subjects?", "adding or changing os subjects date"
            ],
            "responses": [
                "Hi! Here are the scheduled dates for Adding or Changing of Subjects:",
                "",
                "First semester: 3rd week of September",
                "Second semester: 4th week of February",
                "Midyear: 4th week of July"
            ]
        },
        {
            "tag": "DroppingofSubjectsDate",
            "patterns": [
                "When is the Dropping of Subjects?", 
                "dropping od subjects date"
            ],
            "responses": [
                "Hi! Here are the scheduled dates for Dropping of Subjects:",
                "",
                "First semester: 2nd week of October",
                "Second semester: Last week of March",
                "Midyear: 4th week of July"             ]
        },
        {
            "tag": "LastDayofFilingGraduationDate",
            "patterns": [
                "When is the last day of filing an Application for Graduation?", "last day of filing an application for graduation date"
            ],
            "responses": [
                "Hi! Here are the scheduled date for Last Day to file an Application for Graduation:",
                "",
                "First semester: 2nd week of October",
                "Second semester: 3rd week of March",
                "Midyear: last week of July"            ]
        },
        {
            "tag": "StudentWeekDate",
            "patterns": [
                "When is the student week?", "student week date"
            ],
            "responses": [
                "Hi! Student week will occur on the 3rd week of April 2024. It will be held for 3 days."
            ]
        },
        {
            "tag": "ChristmasBreakDate",
            "patterns": [
                "When is the Christmas break or vacation?", "Christmas vacation date"
            ],
            "responses": [
                "Hi! Christmas Vacation is scheduled from 4th week of December, to 1st week of January."
            ]
        },
        {
            "tag": "LastDayofBoundManuscript",
            "patterns": [
                "When is the last day for submission of Bound Manuscript?","last day for submission of bound manuscript date"
            ],
            "responses": [
                "Hi! Here are the scheduled dates for the Last Day for Submission of Bound Manuscript:",
                "",
                "First semester:",
                "----Undergraduate: Last week of January",
                "----Graduate: Last week of January",
                "Second semester:",
                "----Undergraduate: 4th week of June",
                "----Graduate: 4th week of June",
                "Midyear:",
                "----Undergraduate: 4th week of July",
                "----Graduate: 4th week of July"             
            ]
        },
        {
            "tag": "Thanks",
            "patterns": [
                "thanks", "okay"
            ],
            "responses": [
                "You're welcome"
            ]
        },
        {
            "tag": "Goodbye",
            "patterns": [
                "goodbye", "exit"
            ],
            "responses": [
                "Goodbye. Have a great day!"
            ]
        },
        {
            "tag": "Nothing",
            "patterns": [
                "Nothing","None"
            ],
            "responses": [
                "Goodbye. Have a great day!"
            ]
        }
    ]
offensive_words = [
          "fuck", "fucking,",
          "fck", "bitch", "btch","bastard",
          "shit","sht",
          "shitty"
        ]

# Preprocess the intents data  
processed_data = []
intent_tags = {}

# NLTK setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

for idx, intent in enumerate(intents_data):
    examples = intent.get('patterns', [])  # get() handles the missing 'patterns' key

    # Process each example
    for example in examples:
        # Tokenization, lowercasing, removing stop words, stemming, and removing punctuation
        tokens = word_tokenize(example.lower())
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric characters

        # Join tokens back into a sentence
        processed_example = ' '.join(tokens)

        # Store the processed example and its corresponding intent tag
        processed_data.append(processed_example)
        intent_tags[len(processed_data) - 1] = intent.get('tag', '')  # Provide a default value

# Create a bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_data)

# User-specific warning and block dictionary
user_warnings = {}

def rule_based_match(user_input):
    global user_warnings

    # Declare user_id outside the if block
    user_id = hash(user_input)  # A simple way to identify users (replace with a more robust method)

    # Check for offensive words
    if any(word in user_input.lower() for word in offensive_words):

        return "Warning: The use of offensive language is not allowed. Please refrain from using inappropriate words."

    # Tokenize, lowercase, remove stop words, stem, and remove punctuation
    user_tokens = word_tokenize(user_input.lower())
    user_tokens = [token for token in user_tokens if token not in stop_words]
    user_tokens = [stemmer.stem(token) for token in user_tokens]
    user_tokens = [token for token in user_tokens if token.isalnum()]  # Remove non-alphanumeric characters
    processed_user_input = ' '.join(user_tokens)

    # Transform the processed user input using the vectorizer
    user_input_bow = vectorizer.transform([processed_user_input])

    # Find the closest match using a simple rule (cosine similarity)
    similarity_scores = (X @ user_input_bow.T).toarray().flatten()

    # Set a threshold for cosine similarity
    threshold = 0.2
    if max(similarity_scores) >= threshold:
        best_match_index = similarity_scores.argmax()
        return intent_tags.get(best_match_index)

    return None


def get_intent(tag):
    # Implement this function to get the intent based on the tag
    return next((intent for intent in intents_data if intent.get('tag', '') == tag), None)

def get_response(intent):
    # Implement this function to get the response based on the intent
    if intent is not None:
        responses = intent.get('responses', [])
        # Replace newline characters with HTML line break tags
        responses = [response.replace('\n', '<br>') for response in responses]
        return responses
    return None

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data.get('user_input')

    # Perform rule-based matching
    matched_intent_tag = rule_based_match(user_input)

    if matched_intent_tag is not None:
        # Check for offensive words
        if any(word in user_input.lower() for word in offensive_words):
            response = "Warning: The use of offensive language is strictly prohibited. Kindly avoid using inappropriate words to maintain a respectful environment. Repeated violations may result in a temporary ban."
        else:
            # Find the matched intent
            matched_intent = get_intent(matched_intent_tag)

            # Get the responses
            responses = get_response(matched_intent)

            # Combine responses into a single string
            response = "<br>".join(responses) if responses else "I'm sorry, I couldn't find the information you're looking for."

    else:
        response = "I'm sorry, I couldn't find the information you're looking for. Please check your query and try again. If you need further assistance, you can contact our support team."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=False)

