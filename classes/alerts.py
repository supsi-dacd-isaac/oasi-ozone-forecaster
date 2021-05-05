from urllib import request
import json
from slacker import Slacker
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


class SlackClient:

    def __init__(self, logger, cfg):

        self.logger = logger
        self.token = cfg["alerts"]["slack"]["token"]
        self.username = "battery-bot"
        self.icon_url = "https://starbaseatlanta.com/wp-content/uploads/rmhptrs1260.jpg"

        self.channel = cfg["alerts"]["slack"]["channel"]
        self.webhook_url = cfg["alerts"]["slack"]["webHookURL"]

    def send_alert_message(self, msg, color):

        post = {
            "attachments": [
                {
                    "color": color,
                    "text": "{0}".format(msg)
                }
            ],
            "as_user": True,
            "username": self.username,
            "icon_url": self.icon_url
        }
        try:
            self.post(post)
        except Exception as e:
            print(e)

    def send_image(self, img_path):

        slack = Slacker(self.token)
        slack.files.upload(img_path, channels=self.channel)

    def send_info_message(self, msg):

        post = {
                "text": "{0}".format(msg),
                "as_user": False,
                "username": self.username,
                "icon_url": self.icon_url
        }
        self.post(post)

    def send_buttons(self):

        post = {
                  "attachments": [
                    {
                      "fallback": "enjoy ;)",
                      "color": "danger",
                      "actions": [
                        {
                          "type": "button",
                          "text": "Count up to 10...",
                          "url": "https://www.youtube.com/watch?v=gHO9h4CnoIQ&has_verified=1",
                          "style": "danger"
                        },
                        {
                          "type": "button",
                          "text": "Be positive about it",
                          "url": "https://www.youtube.com/watch?v=T70f8SpqMhc",
                          "style": "primary"
                        }
                      ]
                    }
                  ],
                  "as_user": False,
                  "username": self.username,
                  "icon_url": self.icon_url
                }
        self.post(post)

    def post(self, post):

        try:
            json_data = json.dumps(post)
            req = request.Request(url=self.webhook_url,
                                  data=json_data.encode('ascii'),
                                  headers={'Content-Type': 'application/json'})
            resp = request.urlopen(req)
            self.logger.info(f"Sent message to slack.")

        except Exception as e:
            self.logger.error('EXCEPTION: %s' % str(e))


class EmailClient:

    def __init__(self, logger, cfg, subject, test=True):

        self.logger = logger
        self.test = test
        self.cfg = cfg
        if self.test:
            self.to_emails = self.cfg["alerts"]["email"]["receiverAddressesTest"]
        else:
            self.to_emails = self.cfg["alerts"]["email"]["receiverAddresses"]
        self.subject = subject
        self.body = "<!DOCTYPE html>\n<html>\n<body>\n"

    def init_message(self, intro):

        # append intro to email body
        self.body += intro

    def append_to_body(self, text):

        # append text to body
        self.body += text

    def send_email(self):

        self.body += "</body>\n</html>"

        # create Mail instance
        mail = Mail(from_email=self.cfg["alerts"]["email"]["senderAddress"],
                    to_emails=self.cfg["alerts"]["email"]["senderAddress"],
                    subject=self.subject,
                    html_content=self.body)
        for e in self.to_emails:
            mail.add_bcc(bcc_email=e)
        try:
            sg = SendGridAPIClient(
                self.cfg["alerts"]["email"]["sendgridAPIDev01"])
            response = sg.send(mail)
            # self.logger.info(response.body)
            # self.logger.info(response.headers)
            self.logger.info("Sent email to " + ", ".join(self.to_emails))
            self.logger.info(f"Email status code: {response.status_code}")
        except Exception as e:
            self.logger.error('EXCEPTION: %s' % str(e))
