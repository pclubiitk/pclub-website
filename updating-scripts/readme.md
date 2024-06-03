<!-- contributed by the Y23 secys to make the lives of the future secys easier -->

This is the place for scripts that may come in handy when updating the website every year. If anyone makes any new scripts, please document them below.

## secy-csv-to-yaml.py

This is a script to convert the secy details in a csv to yaml which can be directly put in the `contacts.html` file.

The script assumes that there is a CSV file `list.csv` in this format-

```
Name,Hall,Email,Phone No.,Github,Instagram
```

You could get everyone to fill up a Google Sheets file and then directly export the details to a CSV.

Then just run the script with `python secy-csv-to-yaml.py` (or similar, depends on how python is installed) and the yaml will be printed out on the terminal. The yaml will also be written to `secys.yml`
