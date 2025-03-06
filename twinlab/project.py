from pprint import pprint

from typeguard import typechecked

from . import _api, _utils


@typechecked
def list_projects(verbose: bool = False) -> list[str]:
    """List projects that you own or are a part of.

    Projects can be used to group related datasets, emulators, and share them with other users.
    Projects can be created using the ``tl.create_project`` function.

    Args:
        verbose (bool, optional): Display information about the operation while running.

    Returns:
        list: Projects currently available to the user.

    Example:
        .. code-block:: python

            projects = tl.list_projects()
            print(projects)

        .. code-block:: console

            ['biscuits', 'gardening', 'force-energy', 'combusion']

    """
    _, response = _api.get_projects()
    projects = response["projects"]
    projects = [project["name"] for project in projects]
    if verbose:
        print("Projects:")
        pprint(projects, compact=True, sort_dicts=False)
    return projects


@typechecked
def create_project(project_id: str, verbose: bool = False) -> None:
    """Create a new project.

    Projects can be used to group related datasets, emulators, and share them with other users.
    Projects can be shared with other users using the ``tl.share_project`` function.

    Args:
        project_id (str): The name of the project in the twinLab cloud. You cannot create a project with the same id as an existing project.

    Returns:
        None

    """
    _api.post_project(project_id)
    if verbose:
        print(f"Project {project_id} created.")
    return None


@typechecked
def delete_project(project_id: str, verbose: bool = False) -> None:
    """Delete a project that you are the owner of.

    You can only delete a project if you are the owner.

    Args:
        project_id (str): The name of the project in the twinLab cloud.

    Returns:
        None

    """

    # Get the MongoDB id of the project
    # As only the owner can delete the project assume the current user is the owner.
    # If they're not the owner the project will not be found
    _project_id = _utils.get_project_id(project_id, _utils.retrieve_owner(None))

    # Make the delete request
    _api.delete_project(_project_id)
    if verbose:
        print(f"Project {project_id} deleted.")
    return None


@typechecked
def share_project(project_id: str, user: str, role: str, verbose: bool = False) -> None:
    """Share a project with another user.

    You must be the project owner to add users to the project.

    Args:
        project_id (str): The name of the project in the twinLab cloud.
        user (str): The email of the user to share the project with.
        role (str): The role of the user in the project. Can be either "member" or "admin".

    Returns:
        None

    """

    # Get the account object for the user that the project will be shared with
    _, user_account = _api.get_account(user)

    # Get the MongoDB id of the project
    # As only the owner can delete the project assume the current user is the owner.
    # If they're not the owner the project will not be found
    _project_id = _utils.get_project_id(project_id, _utils.retrieve_owner(None))

    # Make the share request
    _api.post_project_members_account(_project_id, user_account["_id"], role)
    if verbose:
        print(f"Project {project_id} shared with user {user}")
    return None


@typechecked
def unshare_project(project_id: str, user: str, verbose: bool = False) -> None:
    """Remove a user from a project.

    You must be the project owner to remove users from the project.

    Args:
        project_id (str): The name of the project in the twinLab cloud.
        user (str): The email of the user to remove from the project.

    Returns:
        None

    """

    # Get the account object for the user that will be removed from the project
    _, user_account = _api.get_account(user)

    # Get the MongoDB id of the project
    # As only the owner can delete the project assume the current user is the owner.
    # If they're not the owner the project will not be found
    _project_id = _utils.get_project_id(project_id, _utils.retrieve_owner(None))

    # Make the unshare request
    _api.delete_project_members_account(_project_id, user_account["_id"])
    if verbose:
        print(f"User {user} removed from project {project_id}")
    return None


@typechecked
def list_project_members(
    project_id: str, project_owner_email: str
) -> dict[str, list[str]]:
    """List members of a project with their project roles.

    Args:
        project_id (str): The name of the project in the twinLab cloud.
        project_owner_email (str): The email of the project owner.

    Returns:
        Dict[str, list[str]]: Members of the project grouped by their role.

    Example:
        .. code-block:: python

            tl.list_project_members("project1", "user1@mail.com")

        .. code-block:: console

            {
                "owner": ["user1@mail.com"],
                "admin": ["user2@mail.com"],
                "member": ["user3@mail.com, "user4@mail.com"]
            }

    """

    # Get the mongoDB id for the project
    project_id_mdb = _utils.get_project_id(project_id, project_owner_email)

    # Get the members of the project
    _, response = _api.get_project_members(project_id_mdb)

    return response
