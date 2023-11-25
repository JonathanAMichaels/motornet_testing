import sys
import os
import smtplib
from datetime import datetime
from os.path import expanduser

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

def send_email(recipient, subject, body, attachment_path):
    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
#    message = """From: %s\nTo: %s\nSubject: %s\n\n%s""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
#    message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)

    msg = MIMEMultipart()
    msg['From'] = FROM
    msg['To'] = ", ".join(TO)
    msg['Subject'] = SUBJECT

    msg.attach(MIMEText(TEXT))

    if (attachment!=""):
        # Attach file
        with open(attachment_path, "rb") as attachment_file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment_file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename=attachment_path.split('/')[-1])
            msg.attach(part)

    message = msg.as_string()

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


if __name__ == "__main__":
    home = expanduser("~")
    f=open(home + "/.gpw","r")
    lines=f.readlines()
    user=lines[0]
    pwd=lines[1]
    f.close()
    subject = f"job {sys.argv[1]} on {os.uname()[1]} has finished"
    body = f"job:\t{sys.argv[1]} \non:\t{os.uname()[1]} \nt1:\t{sys.argv[2]} \nt2:\t{sys.argv[3]}\n"
    if (len(sys.argv) > 3):
        attachment = sys.argv[4]
    else:
        attachment = ""
    send_email("pgribble@uwo.ca", subject, body, attachment)


