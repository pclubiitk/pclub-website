# Pclub Website
Pclub's website is a jekyll based static pages website with a custom theme.

### Dependencies
The only dependency is `jekyll`...which is made with `ruby` so it is and extended dependency. As with anything Ruby, it is recommended to use 

### Running locally
To run the theme locally
- Install `ruby` along with `gem` which probably installs alongside (preferably with a package manager)
- Install `bundle` with `gem install bundle`
- Now clone this repo (fork it if you haven't) and navigate to the cloned directory
- Run `bundle install` to install all the dependencies
- Finally run `bundle exec jekyll serve` to start build the project and start a Jekyll server on localhost
- You can use `bundle exec jekyll serve --livereload` instead to automatically preview the changes as you make them (auto reloads on file save)
- You can preview the site at `localhost:4000` or `127.0.0.1:4000`

It is recommended to check out the official [step-by-step tutorial](https://jekyllrb.com/docs/step-by-step/) which does an excellent job explaining the workings of Jekyll

## Structure of the project
The structure of this project is like any normal jekyll project, with an exception of the `updating-scripts` folder which contains custom scripts to make the task of updating the site every year easier and have nothing to do with the actual site served.

The general structure of a jekyll project is comprihensively explained in [their docs](https://jekyllrb.com/docs/structure/)
