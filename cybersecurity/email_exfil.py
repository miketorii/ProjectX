import smtplib
import time

smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_acct = 'm@gmail.com'
smtp_password = ''
tgt_accts = ['m@gmail.com']

def plain_email(subject, contents):
    print("--------plain email----------")
    message = f'Subject: {subject}\nFrom: {smtp_acct}\n'
    message += f'To: {", ".join(tgt_accts)}\n\n{contents.decode()}'

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_acct, smtp_password)

    server.sendmail(smtp_acct, tgt_acts, message)
    time.sleep(1)
    server.quit()

if __name__ == '__main__':
    print("--------Start--------")

    plain_email("test message subject",b"This is a content of email")
    
    print("--------End--------")    
    
