#!/bin/bash
PACKAGE=obsoper
nosetests --with-coverage \
          --cover-html \
          --cover-package=${PACKAGE} \
          ${PACKAGE}
